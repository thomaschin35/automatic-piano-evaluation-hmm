# Automatic Piano Evaluation (`stft_hmm_v2` & FastAPI Service)

An audio-to-score alignment and diagnostic grading pipeline powered by a timing-first STFT Hidden Markov Model (`stft_hmm_v2`), paired with a FastAPI service (`api.py`) implementing the `schema_version: "1.0"` integration contract for the Piano Project frontend.

---

## Repository Structure

- **`stft_features.py`** — Audio loading, silence trimming, STFT feature extraction, and pitch/onset candidate scoring.
- **`score_models.py`** — Canonical symbolic score definitions (`ode_to_joy`, `c_major_scale`, `fur_elise`, etc.) and beat-level segment slicing.
- **`stft_hmm_v2.py`** — Pitch-gated IOI dynamic tempo tracking, local percentile onset normalization, peak sharpening, Viterbi alignment, and post-alignment note grading.
- **`api.py`** — FastAPI HTTP wrapper exposing `/health`, `/v1/scores`, and `/v1/evaluations`.
- **`eval/`** — Evaluation harness (`eval.cli`), piece specs (`eval/pieces/*.toml`), ground-truth labels (`eval/labels/`), and adapter mappings (`eval/adapters.py`).

---

## Key Algorithmic Updates (`stft_hmm_v2`)

- **Timing-First HMM Alignment (`TimingConfig`):** Decouples expressive tempo drift from pitch errors using `attack_frames = 3` (~70 ms), asymmetric onset floors (`onset_floor = 0.02`, `sustain_onset_floor = 0.12`), bounded wrong-pitch cost (`pitch_floor = 0.50`), and rebalanced emission weights (`onset_weight = 2.5`, `pitch_weight = 1.5`).
- **Local Onset Normalization & NMS Sharpening:** `local_normalize()` scales onset and energy envelopes against a rolling 4-second median/90th-percentile baseline; `sharpen_onsets()` applies non-maximum suppression with a 1-frame halo (`onset_halo = 0.35`) to collapse STFT smears into single attack spikes.
- **Pitch-Gated Dynamic Tempo Curve:** `recording_tempo()` estimates local frames-per-beat via a sliding-window median Inter-Onset Interval (`ioi_notes = 6`) restricted to expected score pitches, ignoring final-note holds (`_drop_final_hold()`).
- **Calibrated Note Grading:** Tuned `present_threshold = 0.32`, `wrong_min_score = 0.35`, `sub_onset_threshold = 0.42`, `repeat_onset_min = 0.25`, and `held_ratio = 2.5` to minimize false alarms on soft dynamics, repeated notes, and phrase endings.

---

## Latest Benchmark Results (102 Takes, 1,478 Notes)

Evaluated across the full 102-take stress-test corpus (`split=all`, run `20261117-213723_v2_6625b8a`):

| Metric | Score |
| :--- | :---: |
| **Overall Error Precision (`err P`)** | **0.57** |
| **Overall Error Recall (`err R`)** | **0.74** |
| **Overall Error $F_1$ (`err F1`)** | **0.64** |
| **Monophonic `wrong_pitch` $F_1$** | **0.59** (0.53 overall) |
| **Monophonic `missed` $F_1$** | **0.45** (0.38 overall) |
| **Played-Pitch Accuracy on Detected Wrong Notes** | **82.4%** |
| **Clean-Take Pass Rate** | **69.6%** (15/31 zero-flag takes) |

**Per-Piece Error $F_1$ Highlights:**
- `canon_lh` (4 takes, 256 notes): **0.80** Error $F_1$ (0.88 P / 0.74 R)
- `c_major_rh` (19 takes, 152 notes): **0.75** Error $F_1$ (0.60 P / 1.00 R; 0.80 `wrong_pitch` $F_1$)
- `ode_melody` (54 takes, 810 notes): **0.68** Error $F_1$ (0.74 P / 0.62 R)
- `e_major_rh` (2 takes, 16 notes): **100.0%** clean-take pass rate (2/2)

---

## Setup & Installation

1. **Install Python dependencies:**
   ```bash
   python -m pip install -r requirements.txt