"""
stft_hmm_v2.py  --  Timing-first STFT-HMM alignment + per-note grading.

Idea: the HMM answers ONLY "when does each score note start/stop?" using
score-independent onset evidence plus a *bounded-penalty* pitch term.
Pitch correctness is judged afterwards, inside the decoded boundaries.
A wrong note therefore can no longer drag the alignment off the score.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import numpy as np

import stft_features
from score_models import ScoreSegment, midi_to_note_name
from stft_hmm import (AlignedSegment, SegmentResult,
                      name_substitutes, print_diagnostic_report, save_alignment_plot)


@dataclass
class TimingConfig:
    attack_frames: int = 3             # wide enough to cover a ~70 ms STFT-smeared transient
    min_dur_fraction: float = 0.40     # a note must last >= this fraction of its recording-scaled length
    max_dur_fraction: float = 1.80     # soft cap, enforced via exit pressure
    onset_floor: float = 0.02          # Bernoulli floor for attack onset evidence
    sustain_onset_floor: float = 0.12  # milder (1-o) penalty so smeared transients don't trap Viterbi
    pitch_floor: float = 0.50          # max cost of a wrong-pitch frame is log(pitch_floor)
    onset_weight: float = 2.5
    pitch_weight: float = 1.5
    lead_in_self_loop: float = 0.6
    local_window_sec: float = 4.0      # rolling window for onset / IOI baseline
    ioi_notes: int = 6                 # median over this many recent inter-onset gaps
    onset_peak_height: float = 0.22
    onset_min_ioi_sec: float = 0.08    # ignore double-triggers closer than this
    # grading
    attack_skip_frames: int = 3
    release_skip_fraction: float = 0.15
    present_threshold: float = 0.40
    present_fraction: float = 0.40
    wrong_min_score: float = 0.40
    sub_onset_threshold: float = 0.5   # substitute must be freshly struck ...
    sub_attack_fraction: float = 0.4   # ... within this leading fraction of the segment


# ---------------------------------------------------------------- observations
def _rolling_percentile(x: np.ndarray, window: int, pct: float) -> np.ndarray:
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1
    pad = window // 2
    xp = np.pad(np.asarray(x, dtype=np.float64), pad, mode="edge")
    views = np.lib.stride_tricks.sliding_window_view(xp, window)
    return np.percentile(views, pct, axis=1)


def local_normalize(x: np.ndarray, window: int) -> np.ndarray:
    """Subtract local median and scale by local 90th percentile so loud/soft passages share a baseline."""
    med = _rolling_percentile(x, window, 50.0)
    hi = _rolling_percentile(x, window, 90.0)
    return np.clip((x - med) / np.maximum(hi - med, 1e-6), 0.0, 1.0)


def _nms_peaks(x: np.ndarray, min_height: float, min_distance: int) -> np.ndarray:
    """Greedy non-maximum suppression: one peak per smear, spaced by min_distance frames."""
    order = np.argsort(x)[::-1]
    taken = np.zeros(x.size, dtype=bool)
    peaks = []
    gap = max(2, min_distance)
    for i in order:
        if x[i] < min_height:
            break
        if taken[i]:
            continue
        peaks.append(int(i))
        taken[max(0, i - gap):min(x.size, i + gap + 1)] = True
    return np.array(sorted(peaks), dtype=int)


def sharpen_onsets(x: np.ndarray, min_height: float, min_distance: int) -> np.ndarray:
    """Keep local maxima (plus a 1-frame halo) so a 5-frame STFT smear does not look like five attacks."""
    peaks = _nms_peaks(x, min_height, min_distance)
    if peaks.size == 0:
        return x
    out = np.zeros_like(x)
    out[peaks] = x[peaks]
    for p in peaks:
        if p > 0:
            out[p - 1] = max(out[p - 1], 0.35 * x[p])
        if p + 1 < x.size:
            out[p + 1] = max(out[p + 1], 0.35 * x[p])
    return out


def onset_any(bundle, cfg: TimingConfig | None = None) -> np.ndarray:
    """Score-independent onset strength in [0,1]: any candidate pitch onset OR energy rise.

    Locally normalized (not global 95th-percentile scaled) and peak-sharpened so fast
    notes are one spike each instead of a 5-frame plateau that Viterbi treats as 'must attack'.
    """
    cfg = cfg or TimingConfig()
    O = np.vstack(list(bundle.candidate_onset_scores.values())).max(axis=0)
    e = bundle.frame_energy
    de = np.maximum(0.0, np.diff(e, prepend=e[0]))
    win = max(21, int(round(cfg.local_window_sec / bundle.audio.seconds_per_frame)))
    f = np.maximum(local_normalize(O, win), 0.5 * local_normalize(de, win))
    min_dist = max(2, int(round(cfg.onset_min_ioi_sec / bundle.audio.seconds_per_frame)))
    return sharpen_onsets(f, cfg.onset_peak_height, min_dist)


def _pitch_gated_onset(bundle) -> np.ndarray:
    """Onset envelope using only score pitches, so pedal thumps don't set the tempo."""
    scores = bundle.note_onset_scores or bundle.candidate_onset_scores
    return np.vstack(list(scores.values())).max(axis=0)


def recording_tempo(bundle, cfg: TimingConfig) -> tuple[float, np.ndarray | None, dict]:
    """Frames-per-beat from a sliding-window median Inter-Onset Interval on pitch-gated peaks.

    Falls back to stretching the score across the whole recording if too few peaks.
    """
    T = bundle.audio.num_frames
    spf = bundle.audio.seconds_per_frame
    total_beats = max(float(bundle.score.total_beats), 1e-6)
    global_fpb = T / total_beats
    info = {"source": "global_stretch", "n_peaks": 0, "bpm": 60.0 / (global_fpb * spf)}

    o = _pitch_gated_onset(bundle)
    win = max(21, int(round(cfg.local_window_sec / spf)))
    o_n = local_normalize(o, win)
    min_dist = max(2, int(round(cfg.onset_min_ioi_sec / spf)))
    peaks = _nms_peaks(o_n, cfg.onset_peak_height, min_dist)
    info["n_peaks"] = int(peaks.size)
    if peaks.size < 4:
        return global_fpb, None, info

    ioi = np.diff(peaks).astype(np.float64)
    med_beats = float(np.median([max(s.duration_beats, 0.25) for s in bundle.segments]))
    k = max(3, min(cfg.ioi_notes, len(ioi)))
    local_ioi = np.array([
        np.median(ioi[max(0, i - k // 2):min(len(ioi), i + k // 2 + 1)])
        for i in range(len(ioi))
    ])
    peak_fpb = local_ioi / med_beats
    t_peaks = peaks[1:].astype(np.float64)
    curve = np.interp(np.arange(T), t_peaks, peak_fpb) if t_peaks.size else np.full(T, global_fpb)
    lo, hi = 0.45 * global_fpb, 1.80 * global_fpb
    curve = np.clip(curve, lo, hi)
    fpb = float(np.clip(np.median(peak_fpb), lo, hi))
    info.update(source="pitch_gated_ioi", bpm=60.0 / (fpb * spf), median_ioi_frames=float(np.median(ioi)))
    return fpb, curve, info


def create_recording_alignment(bundle, cfg: TimingConfig) -> tuple[list[AlignedSegment], dict]:
    """Duration prior from the recording's local IOI curve rather than score BPM × file length."""
    T = bundle.audio.num_frames
    fpb, curve, info = recording_tempo(bundle, cfg)
    durs: list[int] = []
    frame = 0
    for seg in bundle.segments:
        local = float(curve[min(T - 1, frame)]) if curve is not None else fpb
        durs.append(max(3, int(round(seg.duration_beats * local))))
        frame += durs[-1]
    total = sum(durs)
    if total > T:
        scale = T / total
        durs = [max(3, int(round(d * scale))) for d in durs]
        extra = sum(durs) - T
        i = len(durs) - 1
        while extra > 0 and i >= 0:
            take = min(extra, durs[i] - 3)
            durs[i] -= take
            extra -= take
            i -= 1
    alignment = []
    current = 0
    for seg, n in zip(bundle.segments, durs):
        start = min(current, T - 1)
        end = min(T - 1, start + n - 1)
        alignment.append(AlignedSegment(seg, start, end))
        current = end + 1
    return alignment, info


# ---------------------------------------------------------------- state graph
@dataclass(frozen=True)
class TState:
    seg: int          # -1 = lead-in
    kind: str         # 'lead' | 'att' | 'min' | 'free'


def build_graph(alignment: list[AlignedSegment], cfg: TimingConfig):
    states: list[TState] = [TState(-1, "lead")]
    for a in alignment:
        d = a.duration_frames
        n_min = max(1, int(round(cfg.min_dur_fraction * d)) - cfg.attack_frames)
        states += [TState(a.score_segment.index, "att")] * cfg.attack_frames
        states += [TState(a.score_segment.index, "min")] * n_min
        states += [TState(a.score_segment.index, "free")]
    N = len(states)
    A = np.zeros((N, N))
    A[0, 0] = cfg.lead_in_self_loop
    A[0, 1] = 1 - cfg.lead_in_self_loop
    i = 1
    for k, a in enumerate(alignment):
        d = a.duration_frames
        n_min = max(1, int(round(cfg.min_dur_fraction * d)) - cfg.attack_frames)
        first, last = i, i + cfg.attack_frames + n_min          # last == 'free' state
        for j in range(first, last):                              # att + min chain: deterministic
            A[j, j + 1] = 1.0
        mean_free = max(1.5, (cfg.max_dur_fraction + cfg.min_dur_fraction) / 2 * d
                        - cfg.attack_frames - n_min)
        stay = 1.0 - 1.0 / mean_free
        A[last, last] = stay
        if k + 1 < len(alignment):
            A[last, last + 1] = 1.0 - stay
        else:
            A[last, last] = 1.0
        i = last + 1
    return states, A


def log_emissions(states, bundle, cfg: TimingConfig, alignment) -> np.ndarray:
    T = bundle.audio.num_frames
    o = np.clip(onset_any(bundle, cfg), 0, 1)
    e = bundle.frame_energy
    seg_pitch = {a.score_segment.index: a.expected_pitches for a in alignment}
    L = np.zeros((len(states), T))
    eps = cfg.onset_floor
    sus = cfg.sustain_onset_floor
    for n, s in enumerate(states):
        if s.kind == "lead":
            L[n] = np.log(eps + (1 - eps) * (1 - e)) + np.log(eps + (1 - eps) * (1 - o))
            continue
        if s.kind == "att":
            on = np.log(eps + (1 - eps) * o)
            w_p = 0.3                                   # pitch evidence not settled yet
        else:
            on = np.log(sus + (1 - sus) * (1 - o))
            w_p = 1.0
        pit = np.zeros(T)
        for p in seg_pitch[s.seg]:
            sc = bundle.candidate_scores.get(p, np.zeros(T))
            pit += np.log(cfg.pitch_floor + (1 - cfg.pitch_floor) * sc)
        pit /= max(1, len(seg_pitch[s.seg]))
        L[n] = cfg.onset_weight * on + cfg.pitch_weight * w_p * pit
    return L


def viterbi(logE: np.ndarray, A: np.ndarray) -> np.ndarray:
    N, T = logE.shape
    with np.errstate(divide="ignore"):
        logA = np.log(A)
    dp = np.full((N, T), -np.inf); bp = np.zeros((N, T), dtype=int)
    dp[0, 0] = logE[0, 0]; dp[1, 0] = logE[1, 0] + np.log(0.5)   # may start on lead-in or on first attack
    for t in range(1, T):
        c = dp[:, t - 1, None] + logA
        bp[:, t] = np.argmax(c, 0)
        dp[:, t] = np.max(c, 0) + logE[:, t]
    path = np.zeros(T, dtype=int)
    # must finish in the last segment
    path[-1] = N - 1 if np.isfinite(dp[N - 1, -1]) else int(np.argmax(dp[:, -1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[path[t], t]
    return path


def path_to_intervals(path, states, n_seg, T):
    iv = []
    for k in range(n_seg):
        fr = [t for t in range(T) if states[path[t]].seg == k]
        iv.append((fr[0], fr[-1]))
    # lead-in frames get absorbed into segment 0 for display
    iv[0] = (0, iv[0][1])
    iv[-1] = (iv[-1][0], T - 1)
    return iv


# ---------------------------------------------------------------- grading
def grade(intervals, alignment, bundle, cfg: TimingConfig) -> list[SegmentResult]:
    out = []
    for a, (s, e) in zip(alignment, intervals):
        d = e - s + 1
        b0 = min(e, s + cfg.attack_skip_frames)
        b1 = max(b0, e - int(round(cfg.release_skip_fraction * d)))
        body = slice(b0, b1 + 1)
        exp = a.expected_pitches
        detected = tuple(p for p in exp
                         if np.mean(bundle.candidate_scores[p][body] >= cfg.present_threshold)
                         >= cfg.present_fraction)
        subs = name_substitutes(
            bundle, s, e, body, a.score_segment, detected, out[-1].substitutes if out else (),
            onset_threshold=cfg.sub_onset_threshold, attack_fraction=cfg.sub_attack_fraction,
            presence_threshold=cfg.wrong_min_score)
        out.append(SegmentResult(
            segment_index=a.score_segment.index, expected_pitches=exp,
            start_frame=s, end_frame=e,
            expected_start_frame=a.start_frame, expected_end_frame=a.end_frame,
            detected_pitches=detected, has_ghost_wrong=False, substitutes=subs))
    return out


def run_pipeline(audio_path, model_name, out_plot="viterbi_alignment_v2.png"):
    cfg = TimingConfig()
    bundle = stft_features.run_stage1_and_stage2(audio_path, model_name)
    T = bundle.audio.num_frames
    alignment, tempo_info = create_recording_alignment(bundle, cfg)
    print(f"tempo baseline: {tempo_info['source']}  "
          f"{tempo_info['bpm']:.1f} bpm  peaks={tempo_info['n_peaks']}")
    states, A = build_graph(alignment, cfg)
    logE = log_emissions(states, bundle, cfg, alignment)
    path = viterbi(logE, A)
    iv = path_to_intervals(path, states, len(alignment), T)
    results = grade(iv, alignment, bundle, cfg)
    print_diagnostic_report(results, bundle.audio.seconds_per_frame)
    save_alignment_plot(results, alignment, bundle.audio.seconds_per_frame, T, out_plot)
    return bundle, results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp3", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--plot-path", default="viterbi_alignment_v2.png")
    a = ap.parse_args()
    run_pipeline(a.mp3, a.model, a.plot_path)