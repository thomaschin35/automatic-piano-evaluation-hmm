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
from stft_hmm import (AlignedSegment, SegmentResult, create_initial_alignment,
                      name_substitutes, print_diagnostic_report, save_alignment_plot)


@dataclass
class TimingConfig:
    attack_frames: int = 1             # deterministic attack chain length
    min_dur_fraction: float = 0.45     # a note must last >= 45% of its (tempo-scaled) length
    max_dur_fraction: float = 1.80     # soft cap, enforced via exit pressure
    onset_floor: float = 0.02          # Bernoulli floor for onset evidence
    pitch_floor: float = 0.60          # max cost of a wrong-pitch frame is log(pitch_floor)
    onset_weight: float = 3.0
    pitch_weight: float = 1.0
    lead_in_self_loop: float = 0.6
    # grading
    attack_skip_frames: int = 3
    release_skip_fraction: float = 0.15
    present_threshold: float = 0.40
    present_fraction: float = 0.40
    wrong_min_score: float = 0.40
    sub_onset_threshold: float = 0.5   # substitute must be freshly struck ...
    sub_attack_fraction: float = 0.4   # ... within this leading fraction of the segment


# ---------------------------------------------------------------- observations
def onset_any(bundle) -> np.ndarray:
    """Score-independent onset strength in [0,1]: any candidate pitch onset OR energy rise."""
    O = np.vstack(list(bundle.candidate_onset_scores.values())).max(axis=0)
    e = bundle.frame_energy
    de = np.maximum(0.0, np.diff(e, prepend=e[0]))
    de = np.clip(de / max(np.percentile(de, 95), 1e-6), 0, 1)
    f = np.maximum(O, de)
    # +/-1 frame tolerance: onsets are 1-3 frames wide and STFT windows smear them slightly
    pad = np.pad(f, 1, mode="edge")
    return np.maximum.reduce([pad[:-2], pad[1:-1], pad[2:]])


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
    o = np.clip(onset_any(bundle), 0, 1)
    e = bundle.frame_energy
    seg_pitch = {a.score_segment.index: a.expected_pitches for a in alignment}
    L = np.zeros((len(states), T))
    eps = cfg.onset_floor
    for n, s in enumerate(states):
        if s.kind == "lead":
            L[n] = np.log(eps + (1 - eps) * (1 - e)) + np.log(eps + (1 - eps) * (1 - o))
            continue
        if s.kind == "att":
            on = np.log(eps + (1 - eps) * o)
            w_p = 0.3                                   # pitch evidence not settled yet
        else:
            on = np.log(eps + (1 - eps) * (1 - o))
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
    alignment = create_initial_alignment(bundle.segments, bundle.score.total_beats, T)
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