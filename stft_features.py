"""
stage1_stft_features.py (Stage 1 — v2 with Bottom-Up Overtone Suppression)
STFT-based acoustic front-end, frequency-proportional spectral background estimation,
6-harmonic comb fingerprinting, bottom-up overtone cancellation, odd-harmonic bass anchoring,
and pre-sigmoid dB onset flux extraction.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

from score_models import (
    ScoreModel,
    ScoreSegment,
    get_score_model,
    midi_to_note_name,
    slice_score_to_segments,
    print_score_segments,
)


# ---------------------------------------------------------------------------
# Dataclass Configurations
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22_050
    hop_length: int = 512       # ~23.22 ms per frame
    trim_top_db: float = 20.0   # Strips room hiss completely


@dataclass(frozen=True)
class STFTConfig:
    n_fft: int = 4096           # df = 22050 / 4096 = 5.3833 Hz bin spacing
    win_length: int = 4096      # 185.8 ms Hann window: >30 dB isolation between adjacent semitones
    onset_n_fft: int = 2048     # Shorter 92.9 ms window for crisp transient onset flux localization
    window: str = "hann"
    top_db: float = 90.0

    def bin_hz(self, sample_rate: int) -> float:
        return sample_rate / float(self.n_fft)

    def onset_bin_hz(self, sample_rate: int) -> float:
        return sample_rate / float(self.onset_n_fft)

    @property
    def num_bins(self) -> int:
        return self.n_fft // 2 + 1


@dataclass(frozen=True)
class FingerprintConfig:
    # Exact 6 harmonics & weights from mentor flowchart and hmm_em.py
    harmonic_numbers: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    harmonic_weights: tuple[float, ...] = (
        1.00,
        0.90,
        0.70,
        0.50,
        0.35,
        0.25,
    )
    # Standard piano string stiffness coefficient B (stretches upper partials slightly sharp)
    inharmonicity_coeff: float = 0.00015

    # Search window in cents (+/- 30 cents = +/- 0.30 semitones)
    search_tolerance_cents: float = 30.0

    # Frequency-proportional spectral background window (+/- 0.5 octaves around each partial)
    spectral_background_octaves: float = 1.0
    min_background_radius_bins: int = 6

    # Per-frame dynamic range floor below the frame's loudest partial (prevents -80 dB silence
    # floor from giving +50 dB contrast to quiet -30 dB upper harmonics)
    dynamic_floor_margin_db: float = 34.0

    # Sigmoid parameters mapping net contrast (in dB) to [0, 1] pitch evidence
    evidence_midpoint_db: float = 10.0
    evidence_softness_db: float = 3.0

    # Register-aware anchor cutoff:
    # - MIDI >= 52 (E3+): requires fundamental (h1) peak to prevent octave confusion.
    # - MIDI < 52 (Bass): allows odd-harmonic (h2 & h3) rescue when phone mics roll off sub-100 Hz f0.
    bass_anchor_cutoff_midi: int = 52

    # Pre-sigmoid dB spectral flux sigmoid parameters for pitch-specific onsets (works under sustain pedal)
    onset_midpoint_db: float = 5.5
    onset_softness_db: float = 2.0
    onset_min_contrast_db: float = 4.0

    # Bottom-up overtone subtraction: each active lower note predicts its partial levels from its
    # own h1 plus a per-recording partial/h1 ratio over its active frames. A low percentile keeps
    # frames where the upper note is also played (C3+C4, G3+G4) from inflating the ratio; any
    # positive margin erased octaves played at bass loudness. Only energy above the prediction
    # counts toward the upper note.
    partial_prediction_margin_db: float = 0.0
    partial_ratio_percentile: float = 25.0
    partial_ratio_min_frames: int = 10
    default_partial_slope_db_per_octave: float = -6.0
    # Struck C3/G3 measure h2 at -10..-1 dB re h1 (C4 slightly above 0); a "source" whose partials
    # beat its own fundamental by more is a sub-octave ghost (D3 under a played D4 measured +23 dB)
    partial_ratio_max_db: float = 3.0
    source_min_h1_evidence: float = 0.5

    # Candidate pitch bank (E1 = 28 to C7 = 96). Below E3 a semitone is < 1 bin at h1, so bass
    # pitches rely on the h2*h3 anchor rescue (bass_anchor_cutoff_midi) to be told apart.
    candidate_min_midi: int = 28
    candidate_max_midi: int = 96


@dataclass(frozen=True)
class HarmonicComponent:
    harmonic_number: int
    frequency_hz: float
    stft_bin: float
    onset_stft_bin: float
    search_radius_bins: int
    weight: float


@dataclass
class AudioData:
    waveform: np.ndarray
    sample_rate: int
    hop_length: int
    trim_start_sample: int
    trim_end_sample: int
    stft_magnitude: np.ndarray
    stft_db: np.ndarray
    onset_stft_db: np.ndarray

    @property
    def trim_offset_seconds(self) -> float:
        return self.trim_start_sample / float(self.sample_rate)

    @property
    def seconds_per_frame(self) -> float:
        return self.hop_length / float(self.sample_rate)

    @property
    def num_frames(self) -> int:
        return int(self.stft_db.shape[1])


@dataclass
class Stage1FeatureBundle:
    """Complete output of Stage 1 & Stage 2 ready for Stage 3/4 HMM alignment and diagnostic plotting."""
    audio: AudioData
    score: ScoreModel
    segments: list[ScoreSegment]
    candidate_scores: dict[int, np.ndarray]
    candidate_onset_scores: dict[int, np.ndarray]
    note_scores: dict[int, np.ndarray]
    note_onset_scores: dict[int, np.ndarray]
    unexpected_pitch_scores: np.ndarray
    broadband_onset: np.ndarray
    frame_energy: np.ndarray
    observations: np.ndarray
    tracked_pitches: tuple[int, ...]


# ---------------------------------------------------------------------------
# Step 1: Load, Normalize, Trim, and Compute Dual-Resolution STFT
# ---------------------------------------------------------------------------
def midi_to_frequency(midi_note: int) -> float:
    return float(440.0 * (2.0 ** ((midi_note - 69) / 12.0)))


def frequency_to_stft_bin(
    frequency_hz: float,
    sample_rate: int,
    n_fft: int,
) -> float:
    return float(frequency_hz * n_fft / float(sample_rate))


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    wf = np.asarray(waveform, dtype=np.float64)
    centered = wf - np.median(wf)
    peak = np.max(np.abs(centered))
    if peak < 1e-12:
        return np.zeros_like(centered)
    return centered / peak


def _amplitude_to_db(magnitude: np.ndarray, top_db: float = 90.0) -> np.ndarray:
    if _HAS_LIBROSA:
        return librosa.amplitude_to_db(magnitude, ref=np.max, top_db=top_db)
    ref = max(float(np.max(magnitude)), 1e-12)
    db = 20.0 * np.log10(np.maximum(magnitude / ref, 1e-10))
    return np.maximum(db, -float(top_db))


def compute_stft_representations(
    waveform: np.ndarray,
    sample_rate: int,
    audio_config: AudioConfig,
    stft_config: STFTConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _HAS_LIBROSA:
        complex_stft = librosa.stft(
            y=waveform,
            n_fft=stft_config.n_fft,
            win_length=stft_config.win_length,
            hop_length=audio_config.hop_length,
            window=stft_config.window,
            center=True,
        )
        complex_onset_stft = librosa.stft(
            y=waveform,
            n_fft=stft_config.onset_n_fft,
            win_length=stft_config.onset_n_fft,
            hop_length=audio_config.hop_length,
            window=stft_config.window,
            center=True,
        )
    else:
        from scipy.signal import stft as scipy_stft
        _, _, complex_stft = scipy_stft(
            waveform,
            fs=sample_rate,
            window=stft_config.window,
            nperseg=stft_config.win_length,
            noverlap=stft_config.win_length - audio_config.hop_length,
            nfft=stft_config.n_fft,
        )
        _, _, complex_onset_stft = scipy_stft(
            waveform,
            fs=sample_rate,
            window=stft_config.window,
            nperseg=stft_config.onset_n_fft,
            noverlap=stft_config.onset_n_fft - audio_config.hop_length,
            nfft=stft_config.onset_n_fft,
        )

    mag_stft = np.abs(complex_stft)
    stft_db = _amplitude_to_db(mag_stft, top_db=stft_config.top_db)

    mag_onset = np.abs(complex_onset_stft)
    onset_stft_db = _amplitude_to_db(mag_onset, top_db=stft_config.top_db)

    n_frames = stft_db.shape[1]
    if onset_stft_db.shape[1] > n_frames:
        onset_stft_db = onset_stft_db[:, :n_frames]
    elif onset_stft_db.shape[1] < n_frames:
        pad_w = n_frames - onset_stft_db.shape[1]
        onset_stft_db = np.pad(onset_stft_db, ((0, 0), (0, pad_w)), mode="edge")

    return mag_stft, stft_db, onset_stft_db


# In stft_features.py -> preprocess_waveform()
def preprocess_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    audio_config: AudioConfig,
    stft_config: STFTConfig,
) -> AudioData:
    norm_wf = normalize_waveform(waveform)
    
    # Robust trimming: strip leading dead air using 25 dB threshold
    if _HAS_LIBROSA:
        trimmed_wf, bounds = librosa.effects.trim(norm_wf, top_db=25.0)
        trim_start, trim_end = int(bounds[0]), int(bounds[1])
    else:
        trim_start, trim_end = 0, len(norm_wf)
        trimmed_wf = norm_wf

    trimmed_wf = normalize_waveform(trimmed_wf)
    mag_stft, stft_db, onset_stft_db = compute_stft_representations(
        trimmed_wf, sample_rate, audio_config, stft_config
    )
    return AudioData(
        waveform=trimmed_wf,
        sample_rate=sample_rate,
        hop_length=audio_config.hop_length,
        trim_start_sample=trim_start,
        trim_end_sample=trim_end,
        stft_magnitude=mag_stft,
        stft_db=stft_db,
        onset_stft_db=onset_stft_db,
    )


def preprocess_audio(
    audio_path: str | Path,
    audio_config: AudioConfig = AudioConfig(),
    stft_config: STFTConfig = STFTConfig(),
) -> AudioData:
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not _HAS_LIBROSA:
        raise RuntimeError("librosa is required to load audio files from disk.")

    waveform, sr = librosa.load(path, sr=audio_config.sample_rate, mono=True)
    return preprocess_waveform(waveform, sr, audio_config, stft_config)


# ---------------------------------------------------------------------------
# Step 2: Frequency-Proportional & Dynamic-Floor STFT Spectral Background
# ---------------------------------------------------------------------------
def estimate_stft_background(
    stft_db: np.ndarray,
    sample_rate: int,
    stft_config: STFTConfig,
    fingerprint_config: FingerprintConfig,
) -> np.ndarray:
    """
    Estimate the local spectral noise/reverb floor at every STFT bin and frame
    using a frequency-proportional median window (+/- 0.5 octaves around f),
    bounded below by `frame_max_db - dynamic_floor_margin_db` so quiet upper overtones
    do not get inflated against a -80 dB room-silence floor.
    """
    n_bins, n_frames = stft_db.shape
    half_oct = 0.5 * fingerprint_config.spectral_background_octaves
    min_rad = fingerprint_config.min_background_radius_bins

    anchor_bins = np.unique(
        np.clip(
            np.rint(np.geomspace(1.0, n_bins - 1, num=96)).astype(int),
            0,
            n_bins - 1,
        )
    )
    anchor_bg = np.empty((len(anchor_bins), n_frames), dtype=np.float64)

    for idx, k in enumerate(anchor_bins):
        low_k = min(k - min_rad, int(np.floor(k * (2.0 ** (-half_oct)))))
        high_k = max(k + min_rad, int(np.ceil(k * (2.0 ** (+half_oct)))))
        low_k = max(1, low_k)
        high_k = min(n_bins, high_k + 1)
        anchor_bg[idx, :] = np.median(stft_db[low_k:high_k, :], axis=0)

    all_bins = np.arange(n_bins, dtype=np.float64)
    pos = np.searchsorted(anchor_bins, all_bins, side="right") - 1
    pos = np.clip(pos, 0, len(anchor_bins) - 2)
    k0 = anchor_bins[pos]
    k1 = anchor_bins[pos + 1]
    alpha = ((all_bins - k0) / np.maximum(k1 - k0, 1.0))[:, None]

    background_db = (1.0 - alpha) * anchor_bg[pos, :] + alpha * anchor_bg[pos + 1, :]
    frame_max_db = np.max(stft_db, axis=0)
    dynamic_floor_db = frame_max_db[None, :] - fingerprint_config.dynamic_floor_margin_db
    return np.maximum(background_db, dynamic_floor_db)


# ---------------------------------------------------------------------------
# Step 3: 6-Harmonic Fingerprints & Bottom-Up Overtone Suppression
# ---------------------------------------------------------------------------
def build_note_fingerprint(
    midi_note: int,
    sample_rate: int,
    stft_config: STFTConfig,
    fingerprint_config: FingerprintConfig,
) -> tuple[HarmonicComponent, ...]:
    f0 = midi_to_frequency(midi_note)
    B = fingerprint_config.inharmonicity_coeff
    cents_factor = (2.0 ** (fingerprint_config.search_tolerance_cents / 1200.0)) - 1.0

    components: list[HarmonicComponent] = []
    for h, w in zip(
        fingerprint_config.harmonic_numbers,
        fingerprint_config.harmonic_weights,
    ):
        f_h = float(h * f0 * np.sqrt((1.0 + B * (h ** 2)) / (1.0 + B)))
        k_stft = frequency_to_stft_bin(f_h, sample_rate, stft_config.n_fft)
        k_onset = frequency_to_stft_bin(f_h, sample_rate, stft_config.onset_n_fft)

        if 1.0 <= k_stft < stft_config.num_bins - 1:
            radius_bins = int(np.floor(k_stft * cents_factor))
            components.append(
                HarmonicComponent(
                    harmonic_number=h,
                    frequency_hz=f_h,
                    stft_bin=k_stft,
                    onset_stft_bin=k_onset,
                    search_radius_bins=radius_bins,
                    weight=float(w),
                )
            )

    return tuple(components)


def contrast_to_evidence(
    contrast_db: np.ndarray,
    midpoint_db: float,
    softness_db: float,
) -> np.ndarray:
    scaled = np.clip((contrast_db - midpoint_db) / max(softness_db, 1e-6), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def _extract_harmonic_db_and_contrast(
    component: HarmonicComponent,
    stft_db: np.ndarray,
    bin_background: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract raw harmonic energy (dB) and contrast above background (dB).
    Uses exact bracket peak/interpolation at low frequencies (`search_radius_bins == 0`)
    and outer-bracket neighbor comparison (`k0 - 1`, `k1 + 1`) so adjacent semitones cannot leak.
    """
    n_bins, n_frames = stft_db.shape
    if component.search_radius_bins <= 0:
        k_exact = component.stft_bin
        k0 = int(np.floor(k_exact))
        k1 = min(n_bins - 1, k0 + 1)
        frac = k_exact - k0

        # Interpolate and allow peak within the 2 bracketing bins [k0, k1]
        interp_db = (1.0 - frac) * stft_db[k0, :] + frac * stft_db[k1, :]
        bracket_peak_db = np.maximum(stft_db[k0, :], stft_db[k1, :])
        harmonic_db = 0.5 * interp_db + 0.5 * bracket_peak_db
        bg_db = (1.0 - frac) * bin_background[k0, :] + frac * bin_background[k1, :]

        # Check outer neighbors strictly outside [k0, k1] (or +/- 1 around nearest bin)
        k_nearest = int(np.rint(k_exact))
        k_left = max(0, k_nearest - 1)
        k_right = min(n_bins - 1, k_nearest + 1)
        neighbor_max = np.maximum(stft_db[k_left, :], stft_db[k_right, :])
        skirt_penalty = np.maximum(0.0, (neighbor_max - harmonic_db - 2.0) * 2.5)
        contrast_db = (harmonic_db - bg_db) - skirt_penalty
        return harmonic_db, contrast_db

    center_bin = int(np.rint(component.stft_bin))
    first_bin = max(0, center_bin - component.search_radius_bins)
    last_bin = min(n_bins, center_bin + component.search_radius_bins + 1)

    search_band = stft_db[first_bin:last_bin, :]
    rel_idx = np.argmax(search_band, axis=0)
    frame_idx = np.arange(n_frames)
    abs_bins = first_bin + rel_idx

    harmonic_db = search_band[rel_idx, frame_idx]
    bg_db = bin_background[abs_bins, frame_idx]
    contrast_db = harmonic_db - bg_db
    return harmonic_db, contrast_db


def _partial_ratio_profile(
    h_db: np.ndarray,
    active: np.ndarray,
    fingerprint_config: FingerprintConfig,
    max_partial: int,
) -> np.ndarray:
    """
    Level of partial m relative to h1 (dB) for one source note, indexed by partial number
    (index 0 unused, ratio[1] == 0). Measured partials use a robust percentile over the frames
    where the note is active, so the recording's own mic/room tilt is calibrated out; partials
    beyond the fingerprint are extrapolated with a non-increasing log2(m) fit.
    """
    n_h = h_db.shape[0]
    m_all = np.arange(max_partial + 1, dtype=np.float64)
    default = fingerprint_config.default_partial_slope_db_per_octave * np.log2(np.maximum(m_all, 1.0))
    ratio = default.copy()
    if n_h < 2 or int(np.sum(active)) < fingerprint_config.partial_ratio_min_frames:
        return ratio

    for m in range(2, n_h + 1):
        ratio[m] = min(fingerprint_config.partial_ratio_max_db, float(np.percentile(
            h_db[m - 1, active] - h_db[0, active],
            fingerprint_config.partial_ratio_percentile,
        )))

    measured_m = np.arange(2, n_h + 1, dtype=np.float64)
    if len(measured_m) >= 2:
        slope, intercept = np.polyfit(np.log2(measured_m), ratio[2:n_h + 1], 1)
        slope = min(float(slope), 0.0)
        tail = intercept + slope * np.log2(m_all[n_h + 1:])
        ratio[n_h + 1:] = np.minimum(tail, ratio[n_h])
    else:
        ratio[n_h + 1:] = ratio[n_h] + default[n_h + 1:] - default[n_h]
    return ratio


def calculate_all_candidate_features(
    audio: AudioData,
    stft_config: STFTConfig,
    fingerprint_config: FingerprintConfig,
    candidate_pitches: tuple[int, ...] | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Compute STFT presence and pre-sigmoid onset scores across all candidate pitches
    using Bottom-Up Overtone Subtraction:
      - Evaluates pitches from lowest to highest.
      - Whenever a lower pitch `p` is active, predicts the power of its partials at the
        harmonics of the notes on its harmonic intervals (+12, +19, +24, +28, +31, +36).
        Predictions from all lower notes add in linear power and are subtracted from the
        target's observed power, so a lone bass overtone is removed while a struck note on
        that overtone (e.g. G4 over C3 or G3) keeps the energy the bass cannot explain.
    """
    if candidate_pitches is None:
        pitches = tuple(
            sorted(
                range(
                    fingerprint_config.candidate_min_midi,
                    fingerprint_config.candidate_max_midi + 1,
                )
            )
        )
    else:
        pitches = tuple(sorted(candidate_pitches))

    n_frames = audio.num_frames
    n_onset_bins = audio.onset_stft_db.shape[0]

    bin_background = estimate_stft_background(
        audio.stft_db,
        audio.sample_rate,
        stft_config,
        fingerprint_config,
    )

    # Pass 1: Extract raw harmonic dB, raw contrast, and pre-sigmoid dB onset flux per pitch
    fps: dict[int, tuple[HarmonicComponent, ...]] = {}
    raw_h_db: dict[int, np.ndarray] = {}
    raw_c_db: dict[int, np.ndarray] = {}
    raw_flux_db: dict[int, np.ndarray] = {}

    for p in pitches:
        fp = build_note_fingerprint(
            p, audio.sample_rate, stft_config, fingerprint_config
        )
        fps[p] = fp
        if not fp:
            raw_h_db[p] = np.zeros((1, n_frames))
            raw_c_db[p] = np.zeros((1, n_frames))
            raw_flux_db[p] = np.zeros((1, n_frames))
            continue

        h_list: list[np.ndarray] = []
        c_list: list[np.ndarray] = []
        f_list: list[np.ndarray] = []

        for comp in fp:
            h_val, c_val = _extract_harmonic_db_and_contrast(
                comp, audio.stft_db, bin_background
            )
            h_list.append(h_val)
            c_list.append(c_val)

            k_on = int(np.clip(np.rint(comp.onset_stft_bin), 0, n_onset_bins - 1))
            k_lo = max(0, k_on - 1)
            k_hi = min(n_onset_bins, k_on + 2)
            h_on_db = np.max(audio.onset_stft_db[k_lo:k_hi, :], axis=0)

            prev_db = np.pad(h_on_db, (2, 0), mode="edge")[:-2]
            pos_jump_db = np.maximum(0.0, h_on_db - prev_db)
            f_list.append(pos_jump_db)

        raw_h_db[p] = np.vstack(h_list)
        raw_c_db[p] = np.vstack(c_list)
        raw_flux_db[p] = np.vstack(f_list)

    # Pass 2: Bottom-Up Predicted-Partial Subtraction & Final Score Calculation
    predicted_power: dict[int, np.ndarray] = {
        p: np.zeros_like(raw_h_db[p]) for p in pitches
    }
    max_partial = 8 * len(fingerprint_config.harmonic_numbers)
    presence_scores: dict[int, np.ndarray] = {}
    onset_scores: dict[int, np.ndarray] = {}

    for p in pitches:
        fp = fps[p]
        if not fp:
            presence_scores[p] = np.zeros(n_frames)
            onset_scores[p] = np.zeros(n_frames)
            continue

        n_h_p = len(fp)
        w_arr = np.asarray([c.weight for c in fp], dtype=np.float64)

        # Net harmonic contrast after removing the power lower notes' partials explain:
        # lift = -10 log10(1 - predicted / observed), capped at 40 dB when fully explained
        obs_power = 10.0 ** (raw_h_db[p] / 10.0)
        explained = np.clip(predicted_power[p] / np.maximum(obs_power, 1e-30), 0.0, 1.0 - 1e-4)
        eff_c_db = raw_c_db[p] + 10.0 * np.log10(1.0 - explained)
        ev_matrix = contrast_to_evidence(
            eff_c_db,
            midpoint_db=fingerprint_config.evidence_midpoint_db,
            softness_db=fingerprint_config.evidence_softness_db,
        )
        weighted_presence = np.average(ev_matrix, axis=0, weights=w_arr)

        e_h1 = ev_matrix[0, :]
        if p >= fingerprint_config.bass_anchor_cutoff_midi:
            anchor_score = e_h1
        else:
            e_h2 = ev_matrix[1, :] if n_h_p > 1 else np.zeros(n_frames)
            e_h3 = ev_matrix[2, :] if n_h_p > 2 else np.zeros(n_frames)
            odd_bass_rescue = np.sqrt(e_h2 * e_h3)
            anchor_score = np.maximum(e_h1, odd_bass_rescue)

        pres = weighted_presence * anchor_score
        presence_scores[p] = pres

        # Gate onset flux by net post-mask harmonic contrast so overtone ghosts don't trigger onsets
        active_harmonic_gate = (eff_c_db >= fingerprint_config.onset_min_contrast_db).astype(np.float64)
        gated_flux_db = raw_flux_db[p] * active_harmonic_gate
        weighted_flux_db = np.average(gated_flux_db, axis=0, weights=w_arr)
        raw_onset = contrast_to_evidence(
            weighted_flux_db,
            midpoint_db=fingerprint_config.onset_midpoint_db,
            softness_db=fingerprint_config.onset_softness_db,
        )
        onset_scores[p] = raw_onset * np.sqrt(np.maximum(anchor_score, 0.0))

        # Propagate `p`'s predicted partial power upward: harmonic h of target `p + off`
        # sits on partial (mult_m * h) of `p`, predicted from `p`'s h1 level and ratio profile.
        # A source also needs its own fundamental: a bass note found only through the h2*h3
        # rescue may be the virtual fundamental of a fifth (G3 + D4 implies G2) and must not
        # erase the notes that imply it.
        active_note_gate = (pres > 0.25) & (e_h1 > fingerprint_config.source_min_h1_evidence)
        if not np.any(active_note_gate):
            continue
        ratio_db = _partial_ratio_profile(
            raw_h_db[p], active_note_gate, fingerprint_config, max_partial
        )
        src_h1_db = raw_h_db[p][0, :] + fingerprint_config.partial_prediction_margin_db
        gate = active_note_gate.astype(np.float64)

        for semitone_off, mult_m in ((12, 2), (19, 3), (24, 4), (28, 5), (31, 6), (36, 8)):
            p_target = p + semitone_off
            if p_target not in predicted_power or len(fps[p_target]) == 0:
                continue
            for h_idx in range(len(fps[p_target])):
                partial = mult_m * (h_idx + 1)
                pred_db = src_h1_db + ratio_db[partial]
                predicted_power[p_target][h_idx, :] += gate * (10.0 ** (pred_db / 10.0))

    return presence_scores, onset_scores


# ---------------------------------------------------------------------------
# Step 4: Global Onset, RMS Energy, & Out-of-Score Feature Extraction
# ---------------------------------------------------------------------------
def calculate_broadband_onset_feature(
    audio: AudioData,
    stft_config: STFTConfig = STFTConfig(),
) -> np.ndarray:
    """
    Broadband spectral flux onset strength restricted to the piano harmonic band (50–4000 Hz)
    with per-frame median noise subtraction so sustained frames sit cleanly near 0.0.
    """
    df = stft_config.onset_bin_hz(audio.sample_rate)
    k_min = max(1, int(round(50.0 / df)))
    k_max = min(audio.onset_stft_db.shape[0], int(round(4000.0 / df)))
    band_db = audio.onset_stft_db[k_min:k_max, :]

    pos_diff = np.maximum(0.0, np.diff(band_db, axis=1, prepend=band_db[:, :1]))
    # Subtract per-frame median bin flux to remove broadband noise floor (~0.25 baseline)
    frame_median = np.median(pos_diff, axis=0, keepdims=True)
    clean_diff = np.maximum(0.0, pos_diff - 1.25 * frame_median)
    flux = np.mean(clean_diff, axis=0)

    scale = max(float(np.percentile(flux, 98.0)), 1e-6)
    return np.clip(flux / scale, 0.0, 1.0)


def calculate_energy_feature(audio: AudioData) -> np.ndarray:
    """Frame RMS energy scaled robustly between 10th and 95th percentiles to [0, 1]."""
    if _HAS_LIBROSA:
        rms = librosa.feature.rms(
            y=audio.waveform,
            frame_length=2048,
            hop_length=audio.hop_length,
        )[0]
        rms = librosa.util.fix_length(rms, size=audio.num_frames)
    else:
        power = np.mean(audio.stft_magnitude ** 2, axis=0)
        rms = np.sqrt(np.maximum(power, 0.0))

    floor = float(np.percentile(rms, 10.0))
    ceil = float(np.percentile(rms, 95.0))
    return np.clip((rms - floor) / max(ceil - floor, 1e-12), 0.0, 1.0)


def calculate_unexpected_pitch_feature(
    candidate_scores: dict[int, np.ndarray],
    score_pitches: set[int],
    num_frames: int,
) -> np.ndarray:
    """
    Out-of-score pitch activity feature in [0, 1] for Stage 4 Ghost States (WRONG / EXTRA).
    Because `candidate_scores` now has bottom-up overtone masking applied, natural overtones
    of played notes are already suppressed. We additionally exclude direct harmonic intervals
    and sub-octave aliases of `score_pitches`.
    """
    excluded_offsets = (0, -24, -19, -12, 12, 19, 24, 28, 31, 34, 36, 38)
    explainable = {
        p + off
        for p in score_pitches
        for off in excluded_offsets
    }
    out_of_score_arrays = [
        scores
        for midi_p, scores in candidate_scores.items()
        if midi_p not in explainable
    ]
    if not out_of_score_arrays:
        return np.zeros(num_frames, dtype=np.float64)
    return np.max(np.vstack(out_of_score_arrays), axis=0)


def build_observation_matrix(
    note_scores: dict[int, np.ndarray],
    note_onset_scores: dict[int, np.ndarray],
    broadband_onset: np.ndarray,
    frame_energy: np.ndarray,
    unexpected_pitch_scores: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    tracked_pitches = tuple(sorted(note_scores.keys()))
    columns: list[np.ndarray] = [note_scores[p] for p in tracked_pitches]
    columns.extend(note_onset_scores[p] for p in tracked_pitches)
    columns.extend([broadband_onset, frame_energy, unexpected_pitch_scores])
    return np.column_stack(columns), tracked_pitches


def run_stage1_and_stage2(
    audio_path: str | Path,
    model_name: str,
    audio_config: AudioConfig = AudioConfig(),
    stft_config: STFTConfig = STFTConfig(),
    fingerprint_config: FingerprintConfig = FingerprintConfig(),
) -> Stage1FeatureBundle:
    audio = preprocess_audio(audio_path, audio_config, stft_config)
    score = get_score_model(model_name)
    segments = slice_score_to_segments(score)

    candidate_scores, candidate_onset_scores = calculate_all_candidate_features(
        audio, stft_config, fingerprint_config
    )
    score_pitch_set = {
        p for seg in segments for p in seg.expected_pitches
    }
    expected_pitches = sorted(score_pitch_set)

    note_scores = {p: candidate_scores[p] for p in expected_pitches}
    note_onset_scores = {p: candidate_onset_scores[p] for p in expected_pitches}

    broadband_onset = calculate_broadband_onset_feature(audio, stft_config)
    frame_energy = calculate_energy_feature(audio)
    unexpected_pitch_scores = calculate_unexpected_pitch_feature(
        candidate_scores, score_pitch_set, audio.num_frames
    )
    observations, tracked_pitches = build_observation_matrix(
        note_scores,
        note_onset_scores,
        broadband_onset,
        frame_energy,
        unexpected_pitch_scores,
    )

    return Stage1FeatureBundle(
        audio=audio,
        score=score,
        segments=segments,
        candidate_scores=candidate_scores,
        candidate_onset_scores=candidate_onset_scores,
        note_scores=note_scores,
        note_onset_scores=note_onset_scores,
        unexpected_pitch_scores=unexpected_pitch_scores,
        broadband_onset=broadband_onset,
        frame_energy=frame_energy,
        observations=observations,
        tracked_pitches=tracked_pitches,
    )


def plot_stage1_diagnostics(
    bundle: Stage1FeatureBundle,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    n_frames = bundle.audio.num_frames
    times = np.arange(n_frames) * bundle.audio.seconds_per_frame

    pitches = bundle.tracked_pitches
    pitch_labels = [f"{midi_to_note_name(p)} ({p})" for p in pitches]

    pres_mat = np.vstack([bundle.note_scores[p] for p in pitches])
    onset_mat = np.vstack([bundle.note_onset_scores[p] for p in pitches])

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    im0 = axes[0].imshow(
        pres_mat, aspect="auto", origin="lower", interpolation="nearest",
        cmap="magma", vmin=0.0, vmax=1.0,
        extent=[times[0], times[-1], -0.5, len(pitches) - 0.5],
    )
    axes[0].set_yticks(range(len(pitches)))
    axes[0].set_yticklabels(pitch_labels, fontsize=8)
    axes[0].set_ylabel("Score Pitches")
    axes[0].set_title(f"Stage 1 STFT Pitch Presence Scores (Overtone-Suppressed) — {bundle.score.name}")
    fig.colorbar(im0, ax=axes[0], label="Presence [0–1]")

    im1 = axes[1].imshow(
        onset_mat, aspect="auto", origin="lower", interpolation="nearest",
        cmap="viridis", vmin=0.0, vmax=1.0,
        extent=[times[0], times[-1], -0.5, len(pitches) - 0.5],
    )
    axes[1].set_yticks(range(len(pitches)))
    axes[1].set_yticklabels(pitch_labels, fontsize=8)
    axes[1].set_ylabel("Score Pitches")
    axes[1].set_title("Stage 1 Pre-Sigmoid dB Pitch Onset Scores (Detects Pedal Re-Strikes)")
    fig.colorbar(im1, ax=axes[1], label="Onset [0–1]")

    axes[2].plot(times, bundle.frame_energy, label="Frame RMS Energy", color="#2b5c8f", lw=1.5)
    axes[2].plot(times, bundle.broadband_onset, label="Broadband Onset Flux", color="#d95f02", lw=1.2, alpha=0.85)
    axes[2].plot(times, bundle.unexpected_pitch_scores, label="Out-of-Score Pitch Evidence", color="#e7298a", lw=1.2, ls="--")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("Normalized Score")
    axes[2].set_xlabel("Time (seconds, trimmed)")
    axes[2].set_title("Global Energy, Broadband Onset, and Out-of-Score Ghost-State Signal")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(fig)