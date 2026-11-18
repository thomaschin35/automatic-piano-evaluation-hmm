"""
score_models.py
Defines symbolic score data models and score presets for HMM audio-to-score alignment.
"""

from __future__ import annotations
from dataclasses import dataclass
import sys


@dataclass(frozen=True)
class ScoreTrack:
    pitches: list[int]
    durations_beats: list[float]


@dataclass(frozen=True)
class ScoreModel:
    name: str
    melody: ScoreTrack
    harmony: ScoreTrack
    total_beats: float
    bpm_ref: float = 120.0


# ---------------------------------------------------------------------------
# Score Model Builders (Ported from run_pitch_accuracy_cosine.py)
# ---------------------------------------------------------------------------
def _ode_to_joy_with_harmony() -> ScoreModel:
    E_b, Q_b = 1.0, 2.0
    E4, F4, G4, D4, C4, C3, G3 = 64, 65, 67, 62, 60, 48, 55

    mel = [E4, E4, F4, G4, G4, F4, E4, D4, C4, C4, D4, E4, E4, D4, D4]
    mel_beats = [E_b] * 8 + [E_b] * 4 + [1.5 * E_b, 0.5 * E_b, Q_b]

    harm = [C3, 0, G3, 0, C3, 0, G3, 0]
    harm_beats = [2.0 * E_b] * 8

    total_beats = sum(mel_beats)
    return ScoreModel(
        name="ode_to_joy_with_harmony",
        melody=ScoreTrack(pitches=mel, durations_beats=mel_beats),
        harmony=ScoreTrack(pitches=harm, durations_beats=harm_beats),
        total_beats=total_beats,
        bpm_ref=60.0 / 0.44,  # ~136.36
    )


# def _ode_to_joy_without_harmony() -> ScoreModel:
#     E_b, Q_b = 1.0, 2.0
#     E4, F4, G4, D4, C4 = 64, 65, 67, 62, 60

#     mel = [E4, E4, F4, G4, G4, F4, E4, D4, C4, C4, D4, E4, E4, D4, D4]
#     mel_beats = [E_b] * 12 + [1.5 * E_b, 0.5 * E_b, Q_b]

#     total_beats = sum(mel_beats)
#     # No harmony: a single rest spanning the entire melody duration
#     harm = [0]
#     harm_beats = [total_beats]

#     return ScoreModel(
#         name="ode_to_joy_without_harmony",
#         melody=ScoreTrack(pitches=mel, durations_beats=mel_beats),
#         harmony=ScoreTrack(pitches=harm, durations_beats=harm_beats),
#         total_beats=total_beats,
#         bpm_ref=60.0 / 0.44,
#     )
def _ode_to_joy_without_harmony() -> ScoreModel:
    E_b, Q_b = 1.0, 2.0
    # E4=64, F4=65, G4=67, D4=62, C4=60
    E4, F4, G4, D4, C4 = 64, 65, 67, 62, 60

    # The expected score has E4 as the second note:
    mel = [
        E4, E4, F4, G4, G4, F4, E4, D4,
        C4, C4, D4, E4, E4, D4, D4
    ]
    mel_beats = [E_b] * 12 + [1.5 * E_b, 0.5 * E_b, Q_b]

    total_beats = sum(mel_beats)
    harm = [0]
    harm_beats = [total_beats]

    return ScoreModel(
        name="ode_to_joy_without_harmony",
        melody=ScoreTrack(pitches=mel, durations_beats=mel_beats),
        harmony=ScoreTrack(pitches=harm, durations_beats=harm_beats),
        total_beats=total_beats,
        bpm_ref=60.0 / 0.44,
    )


def _fur_elise_with_harmony() -> ScoreModel:
    S_b, E_b = 0.5, 1.0
    E5, Ds5, D5, C5, B4, A4 = 76, 75, 74, 72, 71, 69
    Gs4, E4, C4 = 68, 64, 60

    mel = [
        E5, Ds5, E5, Ds5, E5, B4, D5, C5, A4,
        C4, E4, A4, B4,
        E4, Gs4, B4, C5,
        E4,
        E5, Ds5, E5, Ds5, E5, B4, D5, C5, A4,
        C4, E4, A4, B4,
        E4, C5, B4, A4,
    ]
    mel_beats = [
        S_b, S_b, S_b, S_b, S_b, S_b, S_b, S_b, E_b,
        S_b, S_b, S_b, E_b,
        S_b, S_b, S_b, E_b,
        S_b,
        S_b, S_b, S_b, S_b, S_b, S_b, S_b, S_b, E_b,
        S_b, S_b, S_b, E_b,
        S_b, S_b, S_b, E_b,
    ]

    A2, E3, A3 = 45, 52, 57
    E2, Gs3 = 40, 56

    harm = [
        0,
        A2, E3, A3,
        E2, E3, Gs3,
        A2, E3, A3,
        0,
        A2, E3, A3,
        E2, E3, Gs3, A2,
    ]
    harm_beats = [
        1.76 / 0.44,
        S_b, S_b, 3 * S_b,
        S_b, S_b, 3 * S_b,
        S_b, S_b, 3 * S_b,
        (1.76 - 3 * 0.22) / 0.44,
        S_b, S_b, 3 * S_b,
        S_b, S_b, 3 * S_b, E_b,
    ]

    total_beats = sum(mel_beats)
    return ScoreModel(
        name="fur_elise_with_harmony",
        melody=ScoreTrack(pitches=mel, durations_beats=mel_beats),
        harmony=ScoreTrack(pitches=harm, durations_beats=harm_beats),
        total_beats=total_beats,
        bpm_ref=60.0 / 0.44,
    )


def _fur_elise_without_harmony() -> ScoreModel:
    model = _fur_elise_with_harmony()
    total_beats = model.total_beats
    return ScoreModel(
        name="fur_elise_without_harmony",
        melody=model.melody,
        harmony=ScoreTrack(pitches=[0], durations_beats=[total_beats]),
        total_beats=total_beats,
        bpm_ref=model.bpm_ref,
    )


def _c_major_scale_model() -> ScoreModel:
    # C4, D4, E4, F4, G4, A4, B4, C5
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    # Quarter notes: 1 beat per note, final C5 held for 2 beats
    durations = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    total_beats = sum(durations)

    return ScoreModel(
        name="c_major_scale",
        melody=ScoreTrack(pitches=pitches, durations_beats=durations),
        harmony=ScoreTrack(pitches=[0], durations_beats=[total_beats]),
        total_beats=total_beats,
        bpm_ref=120.0,#120.0,
    )
# def _c_major_scale_model() -> ScoreModel:
#     # C5, D5, E5, F5, G5, A5, B5, C6 (MIDI 72 to 84)
#     pitches = [72, 74, 76, 77, 79, 81, 83, 84]
    
#     # Scale has 8 notes played in ~1.35s (approx 0.16s per note, last note held slightly longer)
#     # Using 1 beat per note, final note 1.5 beats:
#     durations = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5]
#     total_beats = sum(durations)

#     return ScoreModel(
#         name="c_major_scale",
#         melody=ScoreTrack(pitches=pitches, durations_beats=durations),
#         harmony=ScoreTrack(pitches=[0], durations_beats=[total_beats]),
#         total_beats=total_beats,
#         bpm_ref=375.0,  # ~160ms per beat
#     )


# ---------------------------------------------------------------------------
# Registry Lookup
# ---------------------------------------------------------------------------
SCORE_MODELS = {
    "ode_to_joy_with_harmony": _ode_to_joy_with_harmony,
    "ode_to_joy_without_harmony": _ode_to_joy_without_harmony,
    "fur_elise_with_harmony": _fur_elise_with_harmony,
    "fur_elise_without_harmony": _fur_elise_without_harmony,
    "c_major_scale": _c_major_scale_model,
}


def get_score_model(model_name: str) -> ScoreModel:
    if model_name not in SCORE_MODELS:
        raise ValueError(
            f"Unknown model: '{model_name}'. Available models: {list(SCORE_MODELS.keys())}"
        )
    return SCORE_MODELS[model_name]()


# Add to bottom of score_models.py if not already present

@dataclass(frozen=True)
class ScoreSegment:
    index: int
    start_beat: float
    end_beat: float
    melody_pitch: int
    harmony_pitch: int
    onset_pitches: tuple[int, ...]

    @property
    def duration_beats(self) -> float:
        return self.end_beat - self.start_beat

    @property
    def expected_pitches(self) -> tuple[int, ...]:
        pitches = []
        if self.melody_pitch > 0:
            pitches.append(self.melody_pitch)
        if self.harmony_pitch > 0:
            pitches.append(self.harmony_pitch)
        return tuple(sorted(set(pitches)))


def midi_to_note_name(midi_pitch: int) -> str:
    if midi_pitch <= 0:
        return "rest"
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (midi_pitch // 12) - 1
    return f"{names[midi_pitch % 12]}{octave}"


def slice_score_to_segments(score: ScoreModel) -> list[ScoreSegment]:
    melody_beats = 0.0
    mel_intervals = []
    for p, d in zip(score.melody.pitches, score.melody.durations_beats):
        mel_intervals.append((melody_beats, melody_beats + d, p))
        melody_beats += d

    harm_beats = 0.0
    harm_intervals = []
    for p, d in zip(score.harmony.pitches, score.harmony.durations_beats):
        harm_intervals.append((harm_beats, harm_beats + d, p))
        harm_beats += d

    boundaries = {0.0, score.total_beats}
    for s, e, _ in mel_intervals:
        boundaries.add(s); boundaries.add(e)
    for s, e, _ in harm_intervals:
        boundaries.add(s); boundaries.add(e)
    ordered = sorted(boundaries)

    def active_pitch(intervals, b):
        for s, e, p in intervals:
            if s <= b < e:
                return p
        return 0

    segments = []
    for i in range(len(ordered) - 1):
        s_beat, e_beat = ordered[i], ordered[i + 1]
        m_p = active_pitch(mel_intervals, s_beat)
        h_p = active_pitch(harm_intervals, s_beat)
        onsets = tuple(sorted({
            p for s, _, p in (mel_intervals + harm_intervals)
            if p > 0 and abs(s - s_beat) < 1e-4
        }))
        segments.append(ScoreSegment(
            index=i,
            start_beat=s_beat,
            end_beat=e_beat,
            melody_pitch=m_p,
            harmony_pitch=h_p,
            onset_pitches=onsets,
        ))
    return segments


def print_score_segments(segments: list[ScoreSegment]) -> None:
    for seg in segments:
        print(f"Segment {seg.index:2d}: beats {seg.start_beat:5.2f}–{seg.end_beat:5.2f}, expected={seg.expected_pitches}")