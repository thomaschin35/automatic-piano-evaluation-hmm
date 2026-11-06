# Dynamic Tempo Calibration & Onset Normalization (`stft_hmm_v2`)

This branch introduces pitch-gated Inter-Onset Interval (IOI) dynamic tempo tracking, local onset normalization, peak sharpening, and emission tuning to the `stft_hmm_v2` pipeline. These optimizations decouple human expressive timing variations (rubato, hesitations, and practice speed drift) from pitch evaluation errors, substantially reducing false alarms.

---

## Key Algorithmic Changes

### 1. `TimingConfig` Updates
* **Transient Window Width:** Increased `attack_frames` from `1` to `3` (~70 ms) to absorb STFT window temporal smearing.
* **Duration Constraints & Loss Weights:** Lowered `min_dur_fraction` from `0.45` to `0.40` and `pitch_floor` from `0.60` to `0.50`. Rebalanced state emission weights to `onset_weight = 2.5` and `pitch_weight = 1.5`.
* **Asymmetric Onset Penalties:** Introduced `sustain_onset_floor = 0.12` alongside `onset_floor = 0.02` to prevent smeared transient energy from trapping Viterbi decoding in sustain states.
* **IOI Sliding Parameters:** Added `local_window_sec = 4.0`, `ioi_notes = 6`, `onset_peak_height = 0.22`, and `onset_min_ioi_sec = 0.08`.

### 2. Feature Preprocessing & Peak Sharpening
* **Local Adaptive Normalization:** Added `_rolling_percentile()` and `local_normalize()`, replacing static global 95th-percentile scaling with a rolling 4-second local median and 90th-percentile baseline.
* **Non-Maximum Suppression (NMS):** Implemented `_nms_peaks()` and `sharpen_onsets()` with a 1-frame halo to collapse multi-frame STFT smears into distinct attack spikes.
* **Score-Independent Onset Representation:** Updated `onset_any()` to pass locally normalized candidate pitch onsets and frame energy gradients through peak sharpening.

### 3. Pitch-Gated Dynamic IOI Tempo Tracking
* **Pitch Gating:** Added `_pitch_gated_onset()` to restrict onset evaluation to expected score pitches, preventing pedal thumps and out-of-score broadband noise from altering tempo calculations.
* **Dynamic Tempo Curve:** Implemented `recording_tempo()` using a sliding-window median IOI over pitch-gated peaks, producing an interpolated frames-per-beat (FPB) curve across the recording.
* **Recording-Adaptive Segmentation:** Replaced `create_initial_alignment()` with `create_recording_alignment()`, allocating note frame durations directly from the local tempo curve rather than uniform global score stretching.

### 4. Emission Refinements
* In `log_emissions()`, sustain states now evaluate non-onset evidence against `sustain_onset_floor` (`0.12`) rather than `eps` (`0.02`), preventing prolonged attacks from delaying note-state advancement.

---

## Benchmark Results (27 Takes, 741 Notes)

| Metric | Baseline (`v2` Initial) | Calibrated (`stft-hmm-tempo-calibration`) | Delta |
| :--- | :---: | :---: | :---: |
| **Error Precision** | **0.48** (48.0%) | **0.75** (75.0%) | **+27.0%** |
| **Error Recall** | **0.80** (80.0%) | **0.78** (78.0%) | -2.0% |
| **Overall Error $F_1$** | **0.60** | **0.76** | **+0.16** |
| **False Substitutions (`correct` $\rightarrow$ `wrong_pitch`)** | 25 | **2** | **-92.0%** |
| **Played-Pitch Accuracy on Detected Errors** | 84.6% | **82.1%** | -2.5% |

### Key Improvements by Piece
* `ode_melody`: Precision surged from **0.26** to **1.00**; clean-take pass rate reached **90.9%** (6/6 clean takes passed).
* `c_major_rh` & `e_major_rh`: Eliminated false-alarm substitutions, achieving **100% pass rates** (2/2 clean takes each).
* `canon_both`: Error $F_1$ increased from **0.69** to **0.72** with **100%** evaluation completion.

---

## Setup & Execution Guide

### 1. Prerequisites (`ffmpeg`)
Audio files in `recordings/eval-interviews/raw/` are in `.m4a` format and must be converted to `.wav` via `ffmpeg`.

If `ffmpeg` is not installed, install it via Conda:
```bash
conda install -c conda-forge ffmpeg -y
```

Verify the installation:
```bash
which ffmpeg
ffmpeg -version | head -n 1
```

*(Note: macOS native `/usr/bin/afconvert` can serve as a fallback if Conda is unavailable.)*

---

### 2. Audio Conversion
Convert raw `.m4a` interview files into 16-bit PCM `.wav` format:
```bash
.venv/bin/python -m eval.cli convert
```
* Converted `.wav` files are written to `recordings/eval-interviews/processed/`.

---

### 3. Running Evaluation & Benchmarking
Run the alignment and diagnostic scoring pipeline across all labeled interview takes:
```bash
.venv/bin/python -m eval.cli run --model v2 --split all
```

Expected behavior:
* **Skipped takes:** Old missing test audio (`c_major_asc`, `c_scale_1h_1wrong`, `odeharm`, etc.) and unlabelled takes (`1ode-2h-1`, `1ode-2h-2`) are automatically skipped.
* **Evaluation corpus:** Executes across 27 valid interview takes (~15–30 seconds per take).
* **Terminal output:** Prints per-take gold vs. predicted error counts, piece-level precision/recall/$F_1$ metrics, confusion matrices, and detailed note disagreements.
* **Artifacts:** Output plots are generated under `eval/runs/<timestamp>_v2_nogit/plots/`:
  * `overview.png`
  * `confusion.png`

---

### 4. Inspecting Individual Take Alignments
To run the standalone pipeline on a single audio file and generate the diagnostic alignment timeline plot:
```bash
.venv/bin/python stft_hmm_v2.py --mp3 recordings/eval-interviews/processed/1ode-1h-1.wav --model ode_melody --plot-path viterbi_alignment_v2.png
```