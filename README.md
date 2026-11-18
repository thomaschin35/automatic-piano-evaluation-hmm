# Automatic Piano Evaluation (`stft_hmm_v2` & FastAPI Service)

An audio-to-score alignment and diagnostic grading pipeline powered by a timing-first Short-Time Fourier Transform Hidden Markov Model (`stft_hmm_v2`), paired with a versioned FastAPI HTTP service (`api.py`) for integration with the Piano Project frontend.

---

## Architecture & Module Overview

- **`stft_features.py`**: Audio preprocessing (silence trimming, STFT extraction), harmonic pitch candidate scoring, and onset detection.
- **`score_models.py`**: Canonical symbolic score definitions (`ode_to_joy`, `c_major_scale`, `fur_elise`, etc.) and beat-level segment slicing.
- **`stft_hmm_v2.py`**: Pitch-gated Inter-Onset Interval (IOI) dynamic tempo tracking, local percentile onset normalization, peak sharpening, Viterbi state alignment, and post-alignment note grading.
- **`eval/`**: Evaluation harness (`eval.cli`), piece specifications (`eval/pieces/*.toml`), ground-truth labels (`eval/labels/`), and adapter mappings (`eval/adapters.py`).
- **`api.py`**: FastAPI service exposing `/health`, `/v1/scores`, and `/v1/evaluations` conforming to the `schema_version: "1.0"` integration contract.

---

## Key Algorithmic Changes (`stft_hmm_v2`)

### 1. `TimingConfig` & Grading Calibration
- **Transient Window Width:** `attack_frames = 3` (~70 ms) to absorb STFT window temporal smearing.
- **Asymmetric Onset & Pitch Penalties:** `onset_floor = 0.02`, `sustain_onset_floor = 0.12`, and `pitch_floor = 0.50` with `onset_weight = 2.5` and `pitch_weight = 1.5`, keeping wrong-pitch notes aligned to their expected score slot without dragging Viterbi off track.
- **Skip & Repeat Gates:** Tuned `skip_prob = 0.015`, `insert_prob = 0.02`, `repeat_onset_min = 0.35`, and `repeat_missing_penalty = 8.0`.
- **Post-Alignment Grading Thresholds:** Calibrated `present_threshold = 0.32`, `wrong_min_score = 0.35`, `sub_onset_threshold = 0.42`, and `held_ratio = 2.1` to reduce false `missed` and `held_long` alarms on soft dynamics and phrase endings.

### 2. Feature Preprocessing & Peak Sharpening
- **Local Adaptive Normalization:** `_rolling_percentile()` and `local_normalize()` scale onset and energy envelopes against a rolling 4-second local median and 90th percentile so soft and loud passages share a consistent baseline.
- **Non-Maximum Suppression (NMS):** `_nms_peaks()` and `sharpen_onsets()` collapse multi-frame STFT smears into single attack spikes with a 1-frame halo (`onset_halo = 0.35`).

### 3. Pitch-Gated Dynamic IOI Tempo Tracking
- **Pitch Gating:** `_pitch_gated_onset()` restricts tempo peak detection to expected score pitches so pedal thumps and broadband noise do not skew tempo estimation.
- **Dynamic Tempo Curve:** `recording_tempo()` computes a sliding-window median IOI across pitch-gated peaks (`ioi_notes = 6`) to interpolate a local frames-per-beat (FPB) curve, passed into `create_recording_alignment()`.

---

## Benchmark Results

### Expanded Stress-Test Corpus (102 Takes, 1,478 Notes)
Evaluated across all 102 labeled takes including soft/loud dynamics, restarts, pauses, extra notes, and articulation edge cases:

| Metric | Initial 102-Take Baseline | Tuned `stft_hmm_v2` (`shreema-hmm`) | Delta |
| :--- | :---: | :---: | :---: |
| **Error Precision** | 0.40 (40.0%) | **0.57** (57.0%) | **+17.0%** |
| **Error Recall** | **0.78** (78.0%) | 0.74 (74.0%) | -4.0% |
| **Overall Error $F_1$** | 0.53 | **0.64** | **+0.11** |
| **Monophonic `wrong_pitch` $F_1$** | 0.50 | **0.59** | **+0.09** |
| **Overall `missed` $F_1$** | 0.28 | **0.38** | **+0.10** |
| **Played-Pitch Accuracy on Detected Wrong Notes** | 81.2% | **82.4%** | **+1.2%** |
| **Clean-Take Pass Rate** | 9 / 31 (29.0%) | **15 / 31 (48.4%)** | **+19.4%** |

#### Breakdown by Piece (102 Takes)
- **`canon_lh`** (4 takes, 256 notes): **0.88** Precision / **0.79** Recall / **0.83** Error $F_1$
- **`c_major_rh`** (19 takes, 152 notes): **0.60** Precision / **1.00** Recall / **0.75** Error $F_1$ (**0.80** `wrong_pitch` $F_1$)
- **`ode_melody`** (54 takes, 810 notes): **0.74** Precision / **0.62** Recall / **0.68** Error $F_1$
- **`e_major_rh`** (2 takes, 16 notes): **100.0%** clean-take pass rate (2/2)

### Original 27-Take Interview Benchmark (741 Notes)
- **Error Precision:** 0.48 $\rightarrow$ **0.75** (+27.0%)
- **Error Recall:** **0.78**
- **Overall Error $F_1$:** 0.60 $\rightarrow$ **0.76** (+0.16)

---

## Setup & Installation

### 1. Python Dependencies
Install the core audio processing and FastAPI dependencies:
```bash
python -m pip install -r requirements.txt