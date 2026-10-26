"""
score_models.py (Stage 2)
Symbolic score data structures, multi-voice polyphonic segment slicing,
and reference score presets (C Major Scale, Ode to Joy 4m, Moonlight Sonata 4m).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

PitchOrChord = int | tuple[int, ...]


def _normalize_pitches(entry: PitchOrChord) -> tuple[int, ...]:
    """Convert a single MIDI int (0 = rest) or tuple of ints into a sorted tuple of active MIDI pitches."""
    if isinstance(entry, int):
        return (entry,) if entry > 0 else ()
    return tuple(sorted({int(p) for p in entry if p and int(p) > 0}))


@dataclass(frozen=True)
class ScoreTrack:
    """A single musical voice/hand track consisting of sequential events (notes, chords, or rests)."""
    name: str
    pitches: list[PitchOrChord]
    durations_beats: list[float]

    def __post_init__(self) -> None:
        if len(self.pitches) != len(self.durations_beats):
            raise ValueError(
                f"Track '{self.name}' has {len(self.pitches)} pitch entries "
                f"but {len(self.durations_beats)} duration entries."
            )
        if any(d <= 0 for d in self.durations_beats):
            raise ValueError(f"Track '{self.name}' contains non-positive beat durations.")

    @property
    def total_beats(self) -> float:
        return float(sum(self.durations_beats))


@dataclass(frozen=True)
class TrackInterval:
    """A time-bounded event within a single ScoreTrack."""
    track_name: str
    start_beat: float
    end_beat: float
    pitches: tuple[int, ...]


@dataclass(frozen=True)
class ScoreSegment:
    """
    A sliced polyphonic score state spanning [start_beat, end_beat).
    Separates newly struck pitches (onset_pitches) from held-over pitches (sustained_pitches)
    so the HMM can handle polyphonic accompaniment and sustain pedal accurately.
    """
    index: int
    start_beat: float
    end_beat: float
    expected_pitches: tuple[int, ...]
    onset_pitches: tuple[int, ...]
    sustained_pitches: tuple[int, ...]
    measure_number: int
    is_bar_start: bool
    is_phrase_start: bool = False

    @property
    def duration_beats(self) -> float:
        return self.end_beat - self.start_beat


@dataclass(frozen=True)
class ScoreModel:
    """Complete symbolic score container supporting arbitrary polyphonic tracks."""
    name: str
    tracks: tuple[ScoreTrack, ...]
    total_beats: float
    bpm_ref: float = 120.0
    beats_per_measure: float = 4.0
    phrase_start_measures: tuple[int, ...] = (1,)

    @property
    def melody(self) -> ScoreTrack:
        """Backward-compatible accessor for the primary melody/upper track."""
        return self.tracks[0]

    @property
    def harmony(self) -> ScoreTrack:
        """Backward-compatible accessor for the secondary harmony/bass track."""
        if len(self.tracks) > 1:
            return self.tracks[1]
        return ScoreTrack(name="rest", pitches=[0], durations_beats=[self.total_beats])


# ---------------------------------------------------------------------------
# Stage 2: Polyphonic Score Slicing Engine
# ---------------------------------------------------------------------------
def create_track_intervals(track: ScoreTrack) -> list[TrackInterval]:
    """Expand a ScoreTrack into beat-indexed TrackIntervals without cumulative rounding drift."""
    intervals: list[TrackInterval] = []
    current_beat = 0.0
    for entry, duration in zip(track.pitches, track.durations_beats):
        next_beat = current_beat + float(duration)
        intervals.append(
            TrackInterval(
                track_name=track.name,
                start_beat=round(current_beat, 6),
                end_beat=round(next_beat, 6),
                pitches=_normalize_pitches(entry),
            )
        )
        current_beat = next_beat
    return intervals


def slice_score_to_segments(
    score: ScoreModel,
    boundary_tol: float = 1e-5,
) -> list[ScoreSegment]:
    """
    Slice all tracks in a ScoreModel at every unique beat boundary.
    Computes exact `expected_pitches`, `onset_pitches` (newly struck at start_beat),
    `sustained_pitches` (held over from an earlier beat), and bar/phrase anchors.
    """
    all_intervals: list[TrackInterval] = []
    raw_boundaries: list[float] = [0.0, round(score.total_beats, 6)]

    for track in score.tracks:
        track_intervals = create_track_intervals(track)
        all_intervals.extend(track_intervals)
        for iv in track_intervals:
            raw_boundaries.append(iv.start_beat)
            raw_boundaries.append(iv.end_beat)

    # Deduplicate beat boundaries within floating-point tolerance (critical for 1/3 triplets)
    sorted_raw = sorted(raw_boundaries)
    ordered_boundaries: list[float] = [sorted_raw[0]]
    for b in sorted_raw[1:]:
        if abs(b - ordered_boundaries[-1]) > boundary_tol:
            ordered_boundaries.append(b)

    segments: list[ScoreSegment] = []
    for idx in range(len(ordered_boundaries) - 1):
        start_beat = ordered_boundaries[idx]
        end_beat = ordered_boundaries[idx + 1]
        mid_beat = 0.5 * (start_beat + end_beat)

        active_set: set[int] = set()
        onset_set: set[int] = set()

        for iv in all_intervals:
            if iv.start_beat - boundary_tol <= mid_beat < iv.end_beat + boundary_tol:
                active_set.update(iv.pitches)
                if abs(iv.start_beat - start_beat) <= boundary_tol:
                    onset_set.update(iv.pitches)

        expected_pitches = tuple(sorted(active_set))
        onset_pitches = tuple(sorted(onset_set))
        sustained_pitches = tuple(sorted(active_set - onset_set))

        # Bar & phrase structural metadata for Stage 5 Outer-Product Jump Matrix
        beats_per_bar = score.beats_per_measure
        measure_number = int(np.floor((start_beat + boundary_tol) / beats_per_bar)) + 1
        beat_in_bar = (start_beat + boundary_tol) % beats_per_bar
        is_bar_start = beat_in_bar <= 2 * boundary_tol
        is_phrase_start = is_bar_start and (measure_number in score.phrase_start_measures)

        segments.append(
            ScoreSegment(
                index=idx,
                start_beat=start_beat,
                end_beat=end_beat,
                expected_pitches=expected_pitches,
                onset_pitches=onset_pitches,
                sustained_pitches=sustained_pitches,
                measure_number=measure_number,
                is_bar_start=is_bar_start,
                is_phrase_start=is_phrase_start,
            )
        )

    return segments


def midi_to_note_name(midi_note: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    octave = (midi_note // 12) - 1
    return f"{names[midi_note % 12]}{octave}"


def format_pitch_tuple(pitches: tuple[int, ...]) -> str:
    if not pitches:
        return "REST"
    return "+".join(f"{midi_to_note_name(p)}({p})" for p in pitches)


def print_score_segments(segments: list[ScoreSegment]) -> None:
    print(f"{'Seg':>3} | {'Meas':>4} | {'Beats':<11} | {'Dur':>5} | {'Onset Pitches':<24} | {'Sustained Pitches':<22} | {'All Expected Pitches'}")
    print("-" * 105)
    for seg in segments:
        bar_flag = "*" if seg.is_bar_start else " "
        print(
            f"{seg.index:3d} | m.{seg.measure_number:<2d}{bar_flag}| "
            f"{seg.start_beat:5.2f}–{seg.end_beat:5.2f} | "
            f"{seg.duration_beats:5.2f} | "
            f"{format_pitch_tuple(seg.onset_pitches):<24} | "
            f"{format_pitch_tuple(seg.sustained_pitches):<22} | "
            f"{format_pitch_tuple(seg.expected_pitches)}"
        )


# ---------------------------------------------------------------------------
# Score Presets: C Major Scale, Ode to Joy (4m), Moonlight Sonata (4m)
# ---------------------------------------------------------------------------
def _c_major_scale_model() -> ScoreModel:
    """Basic 1-octave ascending C Major Scale: C4, D4, E4, F4, G4, A4, B4, C5."""
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    durations = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]
    total_beats = float(sum(durations))

    return ScoreModel(
        name="c_major_scale",
        tracks=(
            ScoreTrack(name="melody", pitches=pitches, durations_beats=durations),
        ),
        total_beats=total_beats,
        bpm_ref=120.0,
        beats_per_measure=4.0,
        phrase_start_measures=(1,),
    )


def _ode_to_joy_with_harmony() -> ScoreModel:
    """First 4 measures of Beethoven's Ode to Joy with bass harmony."""
    E_b, Q_b = 1.0, 2.0
    E4, F4, G4, D4, C4, C3, G3 = 64, 65, 67, 62, 60, 48, 55

    mel = [E4, E4, F4, G4, G4, F4, E4, D4, C4, C4, D4, E4, E4, D4, D4]
    mel_beats = [E_b] * 12 + [1.5 * E_b, 0.5 * E_b, Q_b]

    harm = [C3, 0, G3, 0, C3, 0, G3, 0]
    harm_beats = [2.0 * E_b] * 8

    total_beats = float(sum(mel_beats))
    return ScoreModel(
        name="ode_to_joy_with_harmony",
        tracks=(
            ScoreTrack(name="melody", pitches=mel, durations_beats=mel_beats),
            ScoreTrack(name="harmony", pitches=harm, durations_beats=harm_beats),
        ),
        total_beats=total_beats,
        bpm_ref=60.0 / 0.44,  # ~136.36 BPM
        beats_per_measure=4.0,
        phrase_start_measures=(1,),
    )


def _ode_to_joy_without_harmony() -> ScoreModel:
    """First 4 measures of Beethoven's Ode to Joy (melody only)."""
    base = _ode_to_joy_with_harmony()
    return ScoreModel(
        name="ode_to_joy_without_harmony",
        tracks=(base.melody,),
        total_beats=base.total_beats,
        bpm_ref=base.bpm_ref,
        beats_per_measure=4.0,
        phrase_start_measures=(1,),
    )


def _moonlight_sonata_4m(use_bass_octaves: bool = True) -> ScoreModel:
    """
    First 4 measures of Beethoven's Piano Sonata No. 14 in C# Minor (Op. 27, No. 2),
    Mvt. 1 (Adagio sostenuto).
    Each measure has 4 quarter-note beats (12 triplet eighth-note subdivisions of 1/3 beat each).
    """
    T_b = 1.0 / 3.0  # 1 triplet eighth-note = 1/3 quarter-note beat
    H_b = 2.0        # Half-note = 2.0 quarter-note beats
    W_b = 4.0        # Whole-note = 4.0 quarter-note beats

    # Right-hand triplet pitches (MIDI)
    Fs3, Gs3, A3, Bs3 = 54, 56, 57, 60  # Note: B#3 is enharmonic to C4 (60)
    Cs4, D4, Ds4, E4, Fs4 = 61, 62, 63, 64, 66

    # Measure 1: 4x [G#3, C#4, E4]
    m1_rh = [Gs3, Cs4, E4] * 4
    # Measure 2: 4x [G#3, C#4, E4]
    m2_rh = [Gs3, Cs4, E4] * 4
    # Measure 3: 2x [A3, C#4, E4] + 2x [A3, D4, F#4]
    m3_rh = [A3, Cs4, E4] * 2 + [A3, D4, Fs4] * 2
    # Measure 4: [G#3, B#3, F#4] + [G#3, C#4, E4] + [G#3, C#4, D#4] + [F#3, B#3, D#4]
    m4_rh = [Gs3, Bs3, Fs4] + [Gs3, Cs4, E4] + [Gs3, Cs4, Ds4] + [Fs3, Bs3, Ds4]

    rh_pitches = m1_rh + m2_rh + m3_rh + m4_rh
    rh_durations = [T_b] * 48  # 48 triplet eighth notes = 16.0 beats

    # Left-hand sustained bass pitches (MIDI)
    if use_bass_octaves:
        lh_pitches: list[PitchOrChord] = [
            (37, 49),  # m.1: C#2 + C#3 (whole note)
            (35, 47),  # m.2: B1 + B2 (whole note)
            (33, 45),  # m.3 (beats 1-2): A1 + A2 (half note)
            (30, 42),  # m.3 (beats 3-4): F#1 + F#2 (half note)
            (32, 44),  # m.4 (beats 1-2): G#1 + G#2 (half note)
            (32, 44),  # m.4 (beats 3-4): G#1 + G#2 (re-struck half note)
        ]
    else:
        lh_pitches = [49, 47, 45, 42, 44, 44]

    lh_durations = [W_b, W_b, H_b, H_b, H_b, H_b]

    return ScoreModel(
        name="moonlight_sonata_4m" if use_bass_octaves else "moonlight_sonata_4m_single_bass",
        tracks=(
            ScoreTrack(name="right_hand_triplets", pitches=rh_pitches, durations_beats=rh_durations),
            ScoreTrack(name="left_hand_bass", pitches=lh_pitches, durations_beats=lh_durations),
        ),
        total_beats=16.0,
        bpm_ref=54.0,  # ~54 quarter-note BPM (~162 triplet subdivisions/min)
        beats_per_measure=4.0,
        phrase_start_measures=(1, 3),
    )


SCORE_MODELS = {
    "c_major_scale": _c_major_scale_model,
    "ode_to_joy_with_harmony": _ode_to_joy_with_harmony,
    "ode_to_joy_without_harmony": _ode_to_joy_without_harmony,
    "moonlight_sonata_4m": lambda: _moonlight_sonata_4m(use_bass_octaves=True),
    "moonlight_sonata_4m_single_bass": lambda: _moonlight_sonata_4m(use_bass_octaves=False),
}


def get_score_model(model_name: str) -> ScoreModel:
    if model_name not in SCORE_MODELS:
        raise ValueError(
            f"Unknown model: '{model_name}'. Available models: {list(SCORE_MODELS.keys())}"
        )
    return SCORE_MODELS[model_name]()
