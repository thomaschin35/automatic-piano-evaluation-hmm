# Piano score-following evaluator: `stft_hmm_v3` (branch `thomas-hmm-2`)

Goal: follow a player through a known score (monophonic lines, simple two-voice pieces) and say,
per note, **correct / wrong pitch / missed / held too long**, with F1 >= 0.9 on monophonic pieces,
and follow the player when they **go back** and try again.

`stft_hmm_v3.py` is a note-level hidden Markov model over detected key strikes. It replaces the
frame-level chain of `stft_hmm_v2.py`, which stays in the repo as the baseline.

## How v3 works

1. **Strikes.** From the unchanged `stft_features.py` front end, a strike is a per-pitch onset peak
   followed by sustained presence of that pitch. Simultaneous strikes form one event; overtones,
   sub-harmonic ghosts and semitone leakage of the strongest pitch are folded into it.
2. **Alignment.** Viterbi over the event sequence; the hidden state is the score position.
   An event can be the next note (right or wrong pitch), a later note (notes in between were
   missed), an earlier note (the player went back), or no note (an extra). The time between
   matched events is compared with the written beat distance at the local tempo, with separate
   allowances for a dropped note that drops its time and an added note that adds its own.
3. **Verdicts.** Read off the path. A wrong key followed by the right one is reported as a wrong
   pitch on that note ("then corrected"). A gap of >= 1.5x its written length is "held too long"
   if the note rang through it and "late" for the next note if it had died away.

```bash
.venv/bin/python stft_hmm_v3.py --audio recordings/eval-interviews/processed/2ode-1h-wrong.wav --model ode_melody
```

## Results

32 labelled takes, 814 notes, two players. "Strict" counts a note only if the label matches
(wrong pitch vs missed); "goal" adds held-too-long. Same test suite for both models.

| Real takes | v2 | v3 |
|---|---|---|
| Monophonic strict F1 (22 takes, 28 error notes) | 0.85 (22 hit / 2 false / 6 missed) | **1.00** (28 / 0 / 0) |
| Monophonic goal F1 (strict + held too long) | 0.80 | **0.93** |
| Monophonic clean takes with no flag | 11 / 12 | **12 / 12** |
| Played pitch named correctly (monophonic) | 95% | 100% |
| Two-hand strict F1 (10 takes) | 0.54 | 0.60 |
| Two-hand clean takes with no flag | 1 / 5 | **5 / 5** |
| All takes strict F1 | 0.69 | 0.81 |

| Synthetic errors made from the 12 clean monophonic takes (fully correct cases) | cases | v2 | v3 |
|---|---|---|---|
| one wrong pitch | 552 | 88% | 99% |
| hand one position off for four notes | 66 | 14% | 88% |
| note missing, different pitch from its neighbour | 240 | 28-48% | 100% |
| note missing, repeat of its neighbour | 126 | 7% | 100% |
| wrong key then the right one / re-struck note | 228 | 0-4% | 97-98% |
| note held through a double-length gap | 126 | 0% | 94% |
| silent double-length gap (next note late) | 126 | 0% | 100% |
| stops before the end | 24 | 8% | 100% |
| goes back to a bar start and plays on (located) | 156 | 0% | 100% |

One real take with a restart exists (`1can-2h-1`, excluded from scoring because the player
"backtracks to the start of measure 5"). v3 reports exactly one restart there, to measure 5 beat 1.

### What these numbers do and do not show

- **The monophonic target is met on the data that exists, and that data is small.** 28 error
  notes come from about ten separate mistakes by two players. The settings were chosen while
  looking at these same takes. Treat 1.00 as "no known failure", not as a measured error rate;
  the synthetic suite and the sensitivity table below are the evidence against overfitting.
- **Held too long is not solved.** None of the 3 monophonic gold labels is hit and one note is
  flagged that is not labelled. The definition works on synthetic cases (94%), but two of the three
  labelled notes measure only 1.0-1.2x their written length, so label and definition disagree.
  "Late" hits 4 of 5 labels and also flags 3 unlabelled gaps of 1.6x-2.4x.
- **Two-voice pieces:** octave scales and Ode with harmony are clean. A right-hand note that
  doubles the left hand at the octave, fifth or third and is *left out* cannot be detected (it is
  indistinguishable from the bass's own overtone on these recordings); a *wrong* one still is.
  `canon_both` remains at about 0.6 strict F1.
- Synthetic cases are edits of real clean takes (score changed, or feature frames cut, repeated,
  reordered). They test the alignment logic on hundreds of cases; they are not new performances.

### Sensitivity of the hand-set numbers

Each of 14 settings moved by -25% and +25%, one at a time (`python -m eval.sensitivity`):

| Outcome over the 28 variants | range |
|---|---|
| Monophonic strict F1 | 0.95 - 1.00 (1.00 in 24 of 28) |
| Monophonic goal F1 | 0.89 - 0.95, except 0.82 with the timing threshold at 1.12x |
| Monophonic clean takes | 12 / 12 in 25 of 28; 11 in two; 9 with the timing threshold at 1.12x |
| All-takes strict F1 | 0.79 - 0.82 |
| Synthetic suite, mean fully-correct | 0.93 - 0.99, except 0.77 with the timing threshold at 1.12x |

The alignment costs (wrong note, skip, extra, timing weight, going back) do not move the real-take
scores at all within +-25%. The settings that matter are the ones that decide what counts as a
strike (`onset_full`, `ghost_ratio`, `extra_report_salience`) and the timing threshold.

## Checking a change

```bash
.venv/bin/python -m eval.cli check --model v3            # unit tests + real takes + synthetic suite
.venv/bin/python -m eval.cli check --model v3 --accept   # store as the accepted scoreboard
.venv/bin/python -m eval.cli run --model v3              # full run with per-note CSV, for `compare`
.venv/bin/python -m eval.sensitivity                     # the table above
```

`check` compares with the accepted scoreboard in `eval/scoreboards/`, lists every metric that moved,
refuses to accept a regression on the goal metrics, and fails if any prediction crashed.
Features are cached under `eval/cache/` (rebuilt when `stft_features.py` changes).
The decisions behind each phase, including the ones that failed, are in [eval/V3_LOG.md](eval/V3_LOG.md).

## Next steps

- Record takes with deliberate mistakes by more players, especially held notes and restarts, and
  agree on what "held too long" means before labelling them.
- Hold out a player: nothing here has been tested on a pianist the settings have not seen.
- Octave doubling needs per-piano calibration (single-hand takes give each key's own overtone
  profile) before a missing doubled note can be detected.

---

# Earlier work on this line: tempo calibration in `stft_hmm_v2`

> Note from the v3 review: replacing the IOI tempo curve below with a plain global stretch changes no
> prediction on these takes, so the gains in this section came from the other changes in that commit
> (wider attack, sustain floor, onset sharpening).

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
.venv/bin/python -m eval.cli run --model v2
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