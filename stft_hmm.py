"""
stft_hmm.py
Integrated STFT-HMM Audio-to-Score Alignment and Diagnostic Grading Pipeline.
Bridges stft_features.py front-end with an outer-product jump HMM and ghost states.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import numpy as np

import stft_features
from score_models import ScoreSegment, midi_to_note_name


# ---------------------------------------------------------------------------
# HMM Configurations & State Definitions
# ---------------------------------------------------------------------------
class NotePhase(str, Enum):
    ATTACK = "attack"
    SUSTAIN = "sustain"
    RELEASE = "release"
    SILENCE = "silence"
    WRONG = "wrong"


@dataclass(frozen=True)
class HMMState:
    index: int
    segment_index: int
    phase: NotePhase
    expected_pitches: tuple[int, ...]
    expected_onset_pitches: tuple[int, ...]
    initial_duration_frames: int
    initial_center_fraction: float


@dataclass
class HMMConfig:
    attack_frames: int = 2
    release_fraction: float = 0.08
    minimum_note_duration_fraction: float = 0.40
    maximum_note_duration_fraction: float = 2.00
    pitch_presence_threshold: float = 0.40
    minimum_presence_fraction: float = 0.30
    position_variance: float = 0.02
    backtrack_jump_epsilon: float = 0.002


@dataclass
class HMMParameters:
    initial_probabilities: np.ndarray
    transition_probabilities: np.ndarray
    emission_means: np.ndarray
    emission_variances: np.ndarray


@dataclass(frozen=True)
class AlignedSegment:
    score_segment: ScoreSegment
    start_frame: int
    end_frame: int

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def expected_pitches(self) -> tuple[int, ...]:
        return self.score_segment.expected_pitches


@dataclass(frozen=True)
class SegmentResult:
    segment_index: int
    expected_pitches: tuple[int, ...]
    start_frame: int
    end_frame: int
    expected_start_frame: int
    expected_end_frame: int
    detected_pitches: tuple[int, ...]
    observed_pitches: tuple[int, ...]
    has_ghost_wrong: bool

    @property
    def all_notes_present(self) -> bool:
        return set(self.expected_pitches).issubset(self.detected_pitches)

    @property
    def is_substitution(self) -> bool:
        # A true substitution requires that the expected notes were NOT found
        return not self.all_notes_present and (bool(self.observed_pitches) or self.has_ghost_wrong)


# ---------------------------------------------------------------------------
# Alignment Setup & State Construction
# ---------------------------------------------------------------------------
def create_initial_alignment(
    segments: list[ScoreSegment],
    total_score_beats: float,
    num_frames: int,
) -> list[AlignedSegment]:
    frames_per_beat = num_frames / total_score_beats

    alignment = []
    current_frame = 0
    for seg in segments:
        seg_frames = max(3, round(seg.duration_beats * frames_per_beat))
        start_f = current_frame
        end_f = min(num_frames - 1, start_f + seg_frames - 1)
        alignment.append(AlignedSegment(seg, start_f, end_f))
        current_frame = end_f + 1

    alignment[-1] = AlignedSegment(alignment[-1].score_segment, alignment[-1].start_frame, num_frames - 1)
    return alignment


def create_hmm_states(
    alignment: list[AlignedSegment],
    config: HMMConfig,
    num_total_frames: int,
) -> list[HMMState]:
    states: list[HMMState] = []
    frame_cursor = 0

    for aligned in alignment:
        seg = aligned.score_segment
        d_frames = aligned.duration_frames
        
        att_frames = max(1, config.attack_frames)
        rel_frames = max(1, round(d_frames * config.release_fraction))
        sus_frames = max(1, d_frames - att_frames - rel_frames)

        phases = [
            (NotePhase.ATTACK, att_frames),
            (NotePhase.SUSTAIN, sus_frames),
            (NotePhase.RELEASE, rel_frames),
            (NotePhase.WRONG, max(1, sus_frames // 2)),
        ]

        for phase, dur in phases:
            center_frac = (frame_cursor + 0.5 * dur) / num_total_frames
            states.append(HMMState(
                index=len(states),
                segment_index=seg.index,
                phase=phase,
                expected_pitches=seg.expected_pitches,
                expected_onset_pitches=seg.onset_pitches,
                initial_duration_frames=dur,
                initial_center_fraction=center_frac,
            ))
            if phase != NotePhase.WRONG:
                frame_cursor += dur

    return states


# ---------------------------------------------------------------------------
# Outer-Product Jump Transition Matrix & Emission Modeling
# ---------------------------------------------------------------------------
def initialize_hmm_parameters(
    states: list[HMMState],
    tracked_pitches: tuple[int, ...],
    config: HMMConfig,
    total_features: int,
) -> HMMParameters:
    N = len(states)
    initial = np.zeros(N)
    initial[0] = 1.0

    A_linear = np.zeros((N, N))
    pitch_to_col = {p: i for i, p in enumerate(tracked_pitches)}
    pitch_onset_to_col = {p: len(tracked_pitches) + i for i, p in enumerate(tracked_pitches)}

    for i, st in enumerate(states):
        stay = max(0.05, 1.0 - (1.0 / max(1.0, float(st.initial_duration_frames))))

        if st.phase == NotePhase.ATTACK:
            A_linear[i, i] = 0.05
            A_linear[i, i + 1] = 0.90
            if i + 3 < N and states[i + 3].phase == NotePhase.WRONG:
                A_linear[i, i + 3] = 0.05
        elif st.phase == NotePhase.SUSTAIN:
            A_linear[i, i] = stay
            A_linear[i, i + 1] = (1.0 - stay) * 0.95
            if i + 2 < N and states[i + 2].phase == NotePhase.WRONG:
                A_linear[i, i + 2] = (1.0 - stay) * 0.05
        elif st.phase == NotePhase.RELEASE:
            A_linear[i, i] = 0.10
            next_seg_att = i + 2
            if next_seg_att < N:
                A_linear[i, next_seg_att] = 0.90
            else:
                A_linear[i, i] = 1.0
        elif st.phase == NotePhase.WRONG:
            A_linear[i, i] = 0.50
            if i - 1 >= 0 and states[i - 1].phase == NotePhase.RELEASE:
                A_linear[i, i - 1] = 0.50
            elif i + 1 < N:
                A_linear[i, i + 1] = 0.50

    # Backtracking outer-product matrix
    # Only allow jumping BACKWARD to earlier attack states
    A_jump = np.zeros((N, N))
    for i, src in enumerate(states):
        if src.phase in (NotePhase.SUSTAIN, NotePhase.RELEASE, NotePhase.WRONG):
            for j, tgt in enumerate(states):
                if tgt.phase == NotePhase.ATTACK and tgt.segment_index <= src.segment_index:
                    A_jump[i, j] = 1.0

    row_sums = A_jump.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    A_jump /= row_sums

    eps = config.backtrack_jump_epsilon
    A = (1.0 - eps) * A_linear + eps * A_jump
    A /= np.maximum(A.sum(axis=1, keepdims=True), 1e-12)

    means = np.full((N, total_features), 0.10)
    variances = np.full((N, total_features), 0.08)

    for st in states:
        idx = st.index
        for p in st.expected_pitches:
            if p in pitch_to_col:
                means[idx, pitch_to_col[p]] = 0.75 if st.phase != NotePhase.RELEASE else 0.40
                variances[idx, pitch_to_col[p]] = 0.05

        if st.phase == NotePhase.ATTACK:
            for p in st.expected_onset_pitches:
                if p in pitch_onset_to_col:
                    means[idx, pitch_onset_to_col[p]] = 0.65
                    variances[idx, pitch_onset_to_col[p]] = 0.02
            means[idx, -3] = 0.65
            variances[idx, -3] = 0.03

        if st.phase == NotePhase.WRONG:
            means[idx, -1] = 0.70
            variances[idx, -1] = 0.04
            for p in st.expected_pitches:
                if p in pitch_to_col:
                    means[idx, pitch_to_col[p]] = 0.10

    return HMMParameters(initial, A, means, variances)


# ---------------------------------------------------------------------------
# Viterbi Sequence Decoding & Boundary Filtering
# ---------------------------------------------------------------------------
def calculate_log_emission_likelihoods(
    observations: np.ndarray,
    parameters: HMMParameters,
) -> np.ndarray:
    diff = observations[None, :, :] - parameters.emission_means[:, None, :]
    vars_ = parameters.emission_variances[:, None, :]
    log_terms = np.log(2.0 * np.pi * vars_) + (diff ** 2) / vars_
    return -0.5 * np.sum(log_terms, axis=2)


def viterbi_decode(
    log_emissions: np.ndarray,
    parameters: HMMParameters,
) -> np.ndarray:
    num_states, num_frames = log_emissions.shape
    log_pi = np.log(np.maximum(parameters.initial_probabilities, 1e-300))
    log_A = np.where(parameters.transition_probabilities > 0,
                     np.log(np.maximum(parameters.transition_probabilities, 1e-300)),
                     -np.inf)

    dp = np.full((num_states, num_frames), -np.inf)
    bp = np.zeros((num_states, num_frames), dtype=int)
    dp[:, 0] = log_pi + log_emissions[:, 0]

    for t in range(1, num_frames):
        candidates = dp[:, t - 1, None] + log_A
        bp[:, t] = np.argmax(candidates, axis=0)
        dp[:, t] = np.max(candidates, axis=0) + log_emissions[:, t]

    path = np.zeros(num_frames, dtype=int)
    path[-1] = int(np.argmax(dp[:, -1]))
    for t in range(num_frames - 1, 0, -1):
        path[t - 1] = bp[path[t], t]

    return path


def smooth_segment_timeline(
    state_path: np.ndarray,
    states: list[HMMState],
    num_segments: int,
) -> list[tuple[int, int]]:
    raw_segments = np.asarray([states[s].segment_index for s in state_path])
    num_frames = len(state_path)

    # Use median filtering over state labels to remove 1-frame spikes/flicker
    kernel_size = 5
    pad = kernel_size // 2
    padded = np.pad(raw_segments, pad, mode="edge")
    filtered = np.asarray([np.median(padded[i : i + kernel_size]) for i in range(num_frames)], dtype=int)

    intervals = []
    for seg_idx in range(num_segments):
        frames = np.flatnonzero(filtered == seg_idx)
        if len(frames) > 0:
            intervals.append((int(frames[0]), int(frames[-1])))
        else:
            prev_end = intervals[-1][1] if intervals else 0
            intervals.append((prev_end, prev_end))

    # Clean gaps and enforce perfectly clamped monotonic progression
    cleaned = []
    for i, (s, e) in enumerate(intervals):
        if i == 0:
            cleaned.append((max(0, s), min(num_frames - 1, e)))
        else:
            prev_e = cleaned[-1][1]
            cur_s = min(num_frames - 1, max(prev_e + 1, s))
            cur_e = min(num_frames - 1, max(cur_s, e))
            cleaned.append((cur_s, cur_e))

    cleaned[-1] = (cleaned[-1][0], num_frames - 1)
    return cleaned


# ---------------------------------------------------------------------------
# Post-Processing & Grading
# ---------------------------------------------------------------------------
def grade_performance(
    state_path: np.ndarray,
    states: list[HMMState],
    alignment: list[AlignedSegment],
    bundle: stft_features.Stage1FeatureBundle,
    config: HMMConfig,
) -> list[SegmentResult]:
    is_wrong_state = np.asarray([states[s].phase == NotePhase.WRONG for s in state_path])
    intervals = smooth_segment_timeline(state_path, states, len(alignment))

    results = []
    for aligned, (start_f, end_f) in zip(alignment, intervals):
        seg_idx = aligned.score_segment.index
        
        # Bulletproof boundary clamps to completely eliminate IndexError
        start_f = max(0, min(start_f, len(state_path) - 1))
        end_f = max(start_f, min(end_f, len(state_path) - 1))
        
        seg_slice = np.arange(start_f, end_f + 1)
        has_wrong = bool(np.any(is_wrong_state[seg_slice])) if len(seg_slice) > 0 else False

        detected = []
        for p in aligned.expected_pitches:
            if p in bundle.note_scores and len(seg_slice) > 0:
                scores = bundle.note_scores[p][seg_slice]
                if np.mean(scores >= config.pitch_presence_threshold) >= config.minimum_presence_fraction:
                    detected.append(p)

        observed = tuple(detected)
        if not set(aligned.expected_pitches).issubset(detected):
            cand_strengths = {
                p: float(np.percentile(sc[seg_slice], 60.0))
                for p, sc in bundle.candidate_scores.items()
            } if len(seg_slice) > 0 else {}
            
            observed = tuple(sorted(
                sorted(cand_strengths, key=cand_strengths.get, reverse=True)[:max(1, len(aligned.expected_pitches))]
            ))

        results.append(SegmentResult(
            segment_index=seg_idx,
            expected_pitches=aligned.expected_pitches,
            start_frame=start_f,
            end_frame=end_f,
            expected_start_frame=aligned.start_frame,
            expected_end_frame=aligned.end_frame,
            detected_pitches=tuple(detected),
            observed_pitches=observed,
            has_ghost_wrong=has_wrong,
        ))

    return results


def print_diagnostic_report(results: list[SegmentResult], seconds_per_frame: float) -> None:
    print("\n" + "=" * 78)
    print("STFT-HMM PERFORMANCE EVALUATION REPORT")
    print("=" * 78)
    for r in results:
        status = "CORRECT"
        if r.is_substitution:
            status = "WRONG NOTE (SUBSTITUTION)"
        elif not r.all_notes_present:
            status = "MISSED NOTE"

        exp_names = [midi_to_note_name(p) for p in r.expected_pitches]
        obs_names = [midi_to_note_name(p) for p in r.observed_pitches]
        t_start = r.start_frame * seconds_per_frame
        t_end = (r.end_frame + 1) * seconds_per_frame
        dur = t_end - t_start

        print(f"Segment {r.segment_index:2d} [{t_start:5.2f}s - {t_end:5.2f}s | {dur:4.2f}s]: "
              f"Expected={exp_names} Observed={obs_names} -> Status: {status}")
    print("=" * 78 + "\n")


def save_alignment_plot(
    results: list[SegmentResult],
    alignment: list[AlignedSegment],
    seconds_per_frame: float,
    num_frames: int,
    output_path: str | Path,
    title: str = "Final duration-constrained Viterbi note alignment",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(14, 3.8))
    expected_color = "#cfd5dd"
    inferred_color = "#67ad7b"
    wrong_color = "#e7a746"
    missing_color = "#df7770"

    # 1. Expected Lane (Bottom)
    for aligned in alignment:
        s_sec = aligned.start_frame * seconds_per_frame
        w_sec = aligned.duration_frames * seconds_per_frame
        axis.broken_barh([(s_sec, w_sec)], (0.15, 0.7), facecolors=expected_color, edgecolors="#4a5568", linewidth=1.0)

    # 2. Inferred Lane (Top)
    for r in results:
        s_sec = r.start_frame * seconds_per_frame
        w_sec = (r.end_frame - r.start_frame + 1) * seconds_per_frame

        if r.is_substitution:
            bar_color = wrong_color
            border_color = "#975a16"
        elif not r.all_notes_present:
            bar_color = missing_color
            border_color = "#9b2c2c"
        else:
            bar_color = inferred_color
            border_color = "#22543d"

        # Explicit contrasting borders cleanly separate sequential green notes
        axis.broken_barh([(s_sec, w_sec)], (1.15, 0.7), facecolors=bar_color, edgecolors=border_color, linewidth=1.2)

        label_x = s_sec + w_sec / 2.0
        pitch_label = "+".join(midi_to_note_name(p) for p in r.observed_pitches) if r.observed_pitches else "rest"
        if w_sec >= 0.12:
            axis.text(label_x, 1.5, pitch_label, ha="center", va="center", fontsize=8.5, weight="semibold", clip_on=True)

    axis.set_yticks([0.5, 1.5], labels=["Expected", "Inferred"])
    axis.set_xlabel("Time (seconds)")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    axis.set_ylim(0.0, 2.05)
    axis.set_xlim(0.0, num_frames * seconds_per_frame)
    axis.legend(
        handles=[
            Patch(facecolor=expected_color, edgecolor="#4a5568", label="Initial score-scaled boundaries"),
            Patch(facecolor=inferred_color, edgecolor="#22543d", label="Current inferred boundaries"),
            Patch(facecolor=wrong_color, edgecolor="#975a16", label="Possible wrong note"),
            Patch(facecolor=missing_color, edgecolor="#9b2c2c", label="Missing / uncertain note"),
        ],
        loc="upper right",
        fontsize=8,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f">> Alignment visualization saved to: {output_path.resolve()}")


# ---------------------------------------------------------------------------
# Main Pipeline Runner
# ---------------------------------------------------------------------------
def run_pipeline(audio_path: str, model_name: str, out_plot: str = "viterbi_alignment.png"):
    print(f">> Stage 1 & 2: Extracting STFT features for {audio_path} [{model_name}]...")
    bundle = stft_features.run_stage1_and_stage2(audio_path, model_name)
    
    hmm_cfg = HMMConfig()
    alignment = create_initial_alignment(bundle.segments, bundle.score.total_beats, bundle.audio.num_frames)
    states = create_hmm_states(alignment, hmm_cfg, bundle.audio.num_frames)
    
    print(f">> Stage 3: Initializing HMM Jump Matrix & Gaussian Emissions ({len(states)} states)...")
    params = initialize_hmm_parameters(states, bundle.tracked_pitches, hmm_cfg, bundle.observations.shape[1])
    
    print(">> Stage 4: Running Viterbi Decoding...")
    log_emissions = calculate_log_emission_likelihoods(bundle.observations, params)
    state_path = viterbi_decode(log_emissions, params)
    
    results = grade_performance(state_path, states, alignment, bundle, hmm_cfg)
    print_diagnostic_report(results, bundle.audio.seconds_per_frame)

    save_alignment_plot(
        results,
        alignment,
        bundle.audio.seconds_per_frame,
        bundle.audio.num_frames,
        out_plot
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run STFT-HMM Audio-to-Score Alignment.")
    parser.add_argument("--mp3", required=True, help="Path to performance recording (.wav or .mp3)")
    parser.add_argument("--model", required=True, help="Score model name (from score_models.py)")
    parser.add_argument("--plot-path", default="viterbi_alignment.png", help="Path to save alignment plot image")
    args = parser.parse_args()

    run_pipeline(args.mp3, args.model, out_plot=args.plot_path)