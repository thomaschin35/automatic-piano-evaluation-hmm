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
    onset_halo: float = 0.35           # sharpened spike: peak plus one frame at this fraction
    skip_prob: float = 0.02            # enter a one-frame skip instead of the note
    insert_prob: float = 0.02          # one extra attack between score notes
    repeat_onset_min: float = 0.35     # a repeated pitch must be struck again
    repeat_missing_penalty: float = 8.0  # nats, first attack frame only
    # grading
    attack_skip_frames: int = 3
    release_skip_fraction: float = 0.15
    present_threshold: float = 0.40
    present_fraction: float = 0.40     # kept for config dumps; presence uses the short window below
    present_window_frames: int = 5
    wrong_min_score: float = 0.40
    sub_onset_threshold: float = 0.5   # substitute must be freshly struck ...
    sub_attack_fraction: float = 0.4   # ... within this leading fraction of the segment
    held_ratio: float = 1.6            # this note's IOI / expected length
    neighbor_ratio: float = 1.35       # neighbors this long mean the tempo changed


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


def sharpen_onsets(x: np.ndarray, min_height: float, min_distance: int, halo: float = 0.35) -> np.ndarray:
    """Keep local maxima (plus a 1-frame halo) so a 5-frame STFT smear does not look like five attacks."""
    peaks = _nms_peaks(x, min_height, min_distance)
    if peaks.size == 0:
        return x
    out = np.zeros_like(x)
    out[peaks] = x[peaks]
    for p in peaks:
        if p > 0:
            out[p - 1] = max(out[p - 1], halo * x[p])
        if p + 1 < x.size:
            out[p + 1] = max(out[p + 1], halo * x[p])
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
    return sharpen_onsets(f, cfg.onset_peak_height, min_dist, cfg.onset_halo)


def _pitch_gated_onset(bundle) -> np.ndarray:
    """Onset envelope using only score pitches, so pedal thumps don't set the tempo."""
    scores = bundle.note_onset_scores or bundle.candidate_onset_scores
    return np.vstack(list(scores.values())).max(axis=0)


def _drop_final_hold(ioi: np.ndarray) -> np.ndarray:
    """Ignore a last gap that is a held final note, not a slower beat."""
    if ioi.size >= 2 and ioi[-1] > 1.8 * float(np.median(ioi[:-1])):
        return ioi[:-1]
    return ioi


def _fit_durations(durs: list[int], limit: int) -> list[int]:
    """Scale every prior by the same factor. Residual frames come off the longest note, not the ending."""
    total = sum(durs)
    if total <= limit:
        return durs
    scale = limit / total
    scaled = [max(3, int(round(d * scale))) for d in durs]
    extra = sum(scaled) - limit
    while extra > 0:
        i = max(range(len(scaled)), key=lambda j: scaled[j])
        if scaled[i] <= 3:
            break
        scaled[i] -= 1
        extra -= 1
    return scaled


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

    ioi = _drop_final_hold(np.diff(peaks).astype(np.float64))
    info["dropped_final_hold"] = int(peaks.size - 1 - ioi.size)
    med_beats = float(np.median([max(s.duration_beats, 0.25) for s in bundle.segments]))
    k = min(len(ioi), max(3, min(cfg.ioi_notes, len(ioi))))
    local_ioi = np.array([
        np.median(ioi[max(0, i - k // 2):min(len(ioi), i + k // 2 + 1)])
        for i in range(len(ioi))
    ])
    peak_fpb = local_ioi / med_beats
    # Ends of the gaps that survived the final-hold drop. Later frames keep that tempo.
    t_peaks = peaks[1:1 + len(ioi)].astype(np.float64)
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
    durs = _fit_durations(durs, T)
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
    seg: int          # -1 = lead-in, -2 = trailing silence
    kind: str         # 'lead' | 'att' | 'min' | 'free' | 'skip' | 'ins' | 'tail'
    repeat_gate: bool = False


def _min_frames(duration: int, cfg: TimingConfig) -> int:
    return max(1, int(round(cfg.min_dur_fraction * duration)) - cfg.attack_frames)


def _same_pitch(a: tuple[int, ...], b: tuple[int, ...] | None) -> bool:
    return bool(a) and a == b


def build_graph(alignment: list[AlignedSegment], cfg: TimingConfig):
    """Left-to-right note chain, plus a skip, one insertion, and a trailing silence.

    A skip bypasses the minimum-duration chain, so a missing note can occupy a
    single ungraded frame and the next onset stays with the next note.
    """
    states: list[TState] = [TState(-1, "lead")]
    skip_at: list[int] = []
    att_at: list[int] = []
    free_at: list[int] = []
    ins_at: list[int] = []
    prev: tuple[int, ...] | None = None
    for a in alignment:
        d = a.duration_frames
        n_min = _min_frames(d, cfg)
        pitches = tuple(a.expected_pitches)
        skip_at.append(len(states))
        states.append(TState(a.score_segment.index, "skip"))
        att_at.append(len(states))
        for j in range(cfg.attack_frames):
            states.append(TState(
                a.score_segment.index, "att",
                repeat_gate=(j == 0 and _same_pitch(pitches, prev)),
            ))
        states += [TState(a.score_segment.index, "min")] * n_min
        free_at.append(len(states))
        states.append(TState(a.score_segment.index, "free"))
        ins_at.append(len(states))
        states.append(TState(a.score_segment.index, "ins"))
        prev = pitches
    tail_at = len(states)
    states.append(TState(-2, "tail"))

    n_seg = len(alignment)
    N = len(states)
    A = np.zeros((N, N))
    p_skip = min(cfg.skip_prob, max(0.0, 1.0 - cfg.lead_in_self_loop))
    A[0, 0] = cfg.lead_in_self_loop
    A[0, skip_at[0]] = p_skip
    A[0, att_at[0]] = 1.0 - cfg.lead_in_self_loop - p_skip

    for k, a in enumerate(alignment):
        d = a.duration_frames
        n_min = _min_frames(d, cfg)
        for j in range(att_at[k], free_at[k] - 1):
            A[j, j + 1] = 1.0
        mean_free = max(1.5, (cfg.max_dur_fraction + cfg.min_dur_fraction) / 2 * d
                        - cfg.attack_frames - n_min)
        stay = 1.0 - 1.0 / mean_free
        p_next_skip = cfg.skip_prob if k + 1 < n_seg else 0.0
        p_ins = cfg.insert_prob
        p_adv = 1.0 - stay - p_next_skip - p_ins
        if p_adv < 0.0:
            stay = max(0.0, 1.0 - p_next_skip - p_ins)
            p_adv = 1.0 - stay - p_next_skip - p_ins
        # The last minimum frame may enter the sustain or leave. Waiting for an
        # extra free frame pushes the next attack off the onset of a short note.
        last_min = free_at[k] - 1
        A[last_min, free_at[k]] = stay
        A[last_min, ins_at[k]] = p_ins
        A[free_at[k], free_at[k]] = stay
        A[free_at[k], ins_at[k]] = p_ins
        if k + 1 < n_seg:
            A[last_min, att_at[k + 1]] = p_adv
            A[last_min, skip_at[k + 1]] = p_next_skip
            A[free_at[k], att_at[k + 1]] = p_adv
            A[free_at[k], skip_at[k + 1]] = p_next_skip
        else:
            A[last_min, tail_at] = p_adv
            A[free_at[k], tail_at] = p_adv

        mean_ins = max(1.5, float(d))
        stay_i = 1.0 - 1.0 / mean_ins
        if k + 1 < n_seg:
            A[ins_at[k], ins_at[k]] = stay_i
            A[ins_at[k], skip_at[k + 1]] = cfg.skip_prob
            A[ins_at[k], att_at[k + 1]] = 1.0 - stay_i - cfg.skip_prob
        else:
            A[ins_at[k], ins_at[k]] = stay_i
            A[ins_at[k], tail_at] = 1.0 - stay_i

        if k + 1 < n_seg:
            A[skip_at[k], skip_at[k + 1]] = cfg.skip_prob
            A[skip_at[k], att_at[k + 1]] = 1.0 - cfg.skip_prob
        else:
            A[skip_at[k], tail_at] = 1.0

    A[tail_at, tail_at] = 1.0
    return states, A


def log_emissions(states, bundle, cfg: TimingConfig, alignment) -> np.ndarray:
    T = bundle.audio.num_frames
    o = np.clip(onset_any(bundle, cfg), 0, 1)
    e = bundle.frame_energy
    seg_pitch = {a.score_segment.index: a.expected_pitches for a in alignment}
    L = np.zeros((len(states), T))
    eps = cfg.onset_floor
    sus = cfg.sustain_onset_floor
    # The sharpened spike is a peak plus a one-frame halo. Treat that whole spike as an attack.
    o_att = np.clip(o / max(cfg.onset_halo, 1e-6), 0.0, 1.0)
    for n, s in enumerate(states):
        if s.kind in ("lead", "tail"):
            L[n] = np.log(eps + (1 - eps) * (1 - e)) + np.log(eps + (1 - eps) * (1 - o_att))
            continue
        if s.kind == "skip":
            # A skip is "this score note did not sound." The transition is the cost
            # when the pitch is absent; hearing the pitch makes the skip a bad frame.
            pit = np.zeros(T)
            for p in seg_pitch[s.seg]:
                sc = bundle.candidate_scores.get(p, np.zeros(T))
                pit += np.log(eps + (1 - eps) * (1 - sc))
            L[n] = cfg.pitch_weight * pit / max(1, len(seg_pitch[s.seg]))
            continue
        if s.kind == "ins":
            # One state has to cover the extra note's attack and its sustain.
            p_att = eps + (1 - eps) * o
            p_sus = sus + (1 - sus) * (1 - o)
            L[n] = cfg.onset_weight * np.log(0.35 * p_att + 0.65 * p_sus)
            continue
        if s.kind == "att":
            on = np.log(eps + (1 - eps) * o_att)
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
        if s.repeat_gate:
            L[n] = L[n] - np.where(o < cfg.repeat_onset_min, cfg.repeat_missing_penalty, 0.0)
    return L


def viterbi(logE: np.ndarray, A: np.ndarray, states: list[TState], cfg: TimingConfig) -> np.ndarray:
    N, T = logE.shape
    with np.errstate(divide="ignore"):
        logA = np.log(A)
    dp = np.full((N, T), -np.inf)
    bp = np.zeros((N, T), dtype=int)
    dp[0, 0] = logE[0, 0]
    skip0 = next(i for i, s in enumerate(states) if s.kind == "skip")
    att0 = next(i for i, s in enumerate(states) if s.kind == "att")
    p_skip = min(cfg.skip_prob, max(0.0, 1.0 - cfg.lead_in_self_loop))
    p_att = max(1e-6, 1.0 - cfg.lead_in_self_loop - p_skip)
    dp[skip0, 0] = logE[skip0, 0] + np.log(max(p_skip, 1e-12))
    dp[att0, 0] = logE[att0, 0] + np.log(p_att)
    for t in range(1, T):
        c = dp[:, t - 1, None] + logA
        bp[:, t] = np.argmax(c, 0)
        dp[:, t] = np.max(c, 0) + logE[:, t]
    last_seg = max(s.seg for s in states)
    end_ids = [
        i for i, s in enumerate(states)
        if s.kind == "tail" or (s.seg == last_seg and s.kind in ("free", "skip", "ins"))
    ]
    finite = [(dp[i, -1], i) for i in end_ids if np.isfinite(dp[i, -1])]
    path = np.zeros(T, dtype=int)
    path[-1] = max(finite)[1] if finite else int(np.argmax(dp[:, -1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[path[t], t]
    return path


def path_to_intervals(path, states, n_seg, T):
    """Graded frames are attack/sustain only. Skips are empty. Insertions are separate."""
    iv = []
    for k in range(n_seg):
        fr = [
            t for t in range(T)
            if states[path[t]].seg == k and states[path[t]].kind in ("att", "min", "free")
        ]
        iv.append((fr[0], fr[-1]) if fr else None)
    insertions = []
    t = 0
    while t < T:
        s = states[path[t]]
        if s.kind == "ins":
            t0, seg = t, s.seg
            while t < T and states[path[t]].kind == "ins" and states[path[t]].seg == seg:
                t += 1
            insertions.append((seg, t0, t - 1))
        else:
            t += 1
    return iv, insertions


# ---------------------------------------------------------------- grading
def _present(scores: np.ndarray, threshold: float) -> bool:
    if scores.size == 0:
        return False
    return float(np.median(scores)) >= threshold


def grade(intervals, alignment, bundle, cfg: TimingConfig) -> list[SegmentResult]:
    out = []
    prev_detected: tuple[int, ...] = ()
    prev_subs: tuple[tuple[int, int], ...] = ()
    for a, iv in zip(alignment, intervals):
        exp = a.expected_pitches
        if iv is None:
            out.append(SegmentResult(
                segment_index=a.score_segment.index, expected_pitches=exp,
                start_frame=-1, end_frame=-1,
                expected_start_frame=a.start_frame, expected_end_frame=a.end_frame,
                detected_pitches=(), has_ghost_wrong=False, substitutes=()))
            continue
        s, e = iv
        b0 = min(e, s + cfg.attack_skip_frames)
        b1 = min(e, b0 + max(1, cfg.present_window_frames) - 1)
        body = slice(b0, b1 + 1)
        detected = tuple(
            p for p in exp
            if _present(bundle.candidate_scores[p][body], cfg.present_threshold)
        )
        sounding = tuple(sorted(set(prev_detected) | {q for _, q in prev_subs}))
        subs = name_substitutes(
            bundle, s, e, body, a.score_segment, detected, prev_subs,
            onset_threshold=cfg.sub_onset_threshold, attack_fraction=cfg.sub_attack_fraction,
            presence_threshold=cfg.wrong_min_score, previous_detected=sounding)
        out.append(SegmentResult(
            segment_index=a.score_segment.index, expected_pitches=exp,
            start_frame=s, end_frame=e,
            expected_start_frame=a.start_frame, expected_end_frame=a.end_frame,
            detected_pitches=detected, has_ghost_wrong=False, substitutes=subs))
        prev_detected, prev_subs = detected, subs
    return out


def align_bundle(bundle, cfg: TimingConfig | None = None):
    """Decode one feature bundle. Returns results, insertions, tempo info, alignment."""
    cfg = cfg or TimingConfig()
    alignment, info = create_recording_alignment(bundle, cfg)
    states, A = build_graph(alignment, cfg)
    logE = log_emissions(states, bundle, cfg, alignment)
    path = viterbi(logE, A, states, cfg)
    intervals, insertions = path_to_intervals(path, states, len(alignment), bundle.audio.num_frames)
    results = grade(intervals, alignment, bundle, cfg)
    return results, insertions, info, alignment


def run_pipeline(audio_path, model_name, out_plot="viterbi_alignment_v2.png"):
    cfg = TimingConfig()
    bundle = stft_features.run_stage1_and_stage2(audio_path, model_name)
    results, insertions, tempo_info, alignment = align_bundle(bundle, cfg)
    print(f"tempo baseline: {tempo_info['source']}  "
          f"{tempo_info['bpm']:.1f} bpm  peaks={tempo_info['n_peaks']}")
    print_diagnostic_report(results, bundle.audio.seconds_per_frame)
    save_alignment_plot(results, alignment, bundle.audio.seconds_per_frame,
                        bundle.audio.num_frames, out_plot)
    return bundle, results, insertions


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp3", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--plot-path", default="viterbi_alignment_v2.png")
    a = ap.parse_args()
    run_pipeline(a.mp3, a.model, a.plot_path)