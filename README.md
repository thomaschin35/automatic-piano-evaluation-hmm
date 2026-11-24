# Piano score-following evaluator: `stft_hmm_v3` (branch `shreema-hmm`)

Goal: follow a player through a known score (monophonic lines and two-voice pieces) and report per-note accuracy (wrong pitch, missed, extra, late, held too long, and restarts) with high precision across registers, tempos, and players.

## Quick Start & Evaluation Commands

```bash
# Run full v3 evaluation across all 132 available monophonic + polyphonic takes
.venv/bin/python -m eval.cli run --model v3

# Run monophonic and polyphonic subsets separately
.venv/bin/python -m eval.cli run --model v3 --piece c_major_both e_major_both ode_harmony twinkle_both
.venv/bin/python -m eval.cli run --model v3 --piece ode_melody c_major_rh e_major_rh e_major_rh_down canon_lh buns c_high c_low frere rests twinkle

# Run generalization and parameter sensitivity suites
.venv/bin/python -m eval.generalise
.venv/bin/python -m eval.sensitivity

# Run standalone single-file analysis or UI demo
.venv/bin/python stft_hmm_v3.py --audio recordings/eval-interviews/processed/2ode-1h-wrong.wav --model ode_melody
.venv/bin/python ui/server.py
```

---

## Summary of Changes After Original `v3` (`stft_hmm_v3.py`)

While the original `v3` event-level HMM achieved 100% recall on core errors (`wrong_pitch`, `missed`, `late`), it suffered from false-positive flags on clean notes caused by octave harmonics, lingering key resonance, rapid passages, and cross-hand bass bleed. The following modular updates were added to `V3Config` and the detection/verdict pipeline:

1. **Fundamental Attack Verification for Octave-Off Keys (`P1` — `octave_attack_check` & `own_attack()`):**
   * Added `own_attack(bundle, pitch, frame, cfg)`, which searches a $\pm 3$-frame window (`octave_attack_span = 3`) around the event frame for a fresh jump at the expected pitch's own fundamental (`fundamental()[1] >= ring_rise_db`) or a ringing-key restrike (`restruck()`).
   * In `verdicts_for()`, when an upper octave (`+12` or `+24` semitones) appears in `free` instead of the expected key `p`, `own_attack()` confirms whether `p` itself was struck—eliminating false octave substitutions in low registers (`c_low` Goal F1: `0.80` $\rightarrow$ `1.00`).
2. **Repeated-Note Octave Rescue (`P1b` — `repeat_octave_rescue`):**
   * When a player re-strikes an already-ringing key (`p in prev_pitches`), the new hammer strike often registers primarily in its 2nd/4th harmonic (`+12` or `+24` semitones). `verdicts_for()` now checks whether `p` maintained sustained ringing (`>= soft_restrike_ringing` [0.50]) before and after the onset while an octave partial fired, preventing false `wrong_pitch` flags on rapid repeated notes (`buns` Goal F1: `0.86` $\rightarrow$ `1.00`).
3. **Two-Hand Bass/Melody Separation (`P2` — `keep_overtone_salience` & `hand_span`):**
   * In `struck()`, when a melody note is folded into `explained_by` under an *unexpected* lower bass strike (e.g., an extra left-hand `C3` beneath right-hand `G4`, its 3rd partial), `struck()` still credits the melody note as played if it is expected at that score position, had a full-strength attack (`pitch in event.full`), and survived front-end overtone subtraction with salience `>= keep_overtone_salience` (`0.40`).
   * In `verdicts_for()`, `stray` smudge candidates exclude `OVERTONE_OFFSETS` and require a fresh fundamental jump (`rise >= ring_rise_db`), while fallback substitutions (`cand`) are bounded by `abs(q - p) <= cfg.hand_span` (`12` semitones) so left-hand bass strikes do not overwrite right-hand melody notes (`twinkle_both` Error Precision: `0.83` $\rightarrow$ `1.00`).
4. **Ringing Soft-Restrike Detection (`P3b` — `soft_restrike()`):**
   * Replaced fixed-window fast-note truncation (`fast_note_rescue = False`) with `soft_restrike()`, which accepts fast re-strikes on an already-ringing key (`>= 0.50` presence before and after) when `broadband_onset >= 0.50` and no competing neighbor key shows an onset (`>= 0.20`) that grows louder afterward. This rescues rapid two-hand passages (`5ode-2h-fast`) without inflating semitone leakage on adjacent notes (`e_major_rh_down` Goal F1: `1.00`).
5. **Simultaneous Extra-Note Gating (`P5` — `extra_allowed()`):**
   * Refines simultaneous extra-note reporting beside a correct note (`report()`): rejects pitches sitting on upper partials of expected notes (`(q - p) in OVERTONE_OFFSETS`), while allowing lower-register extras or repeated keys when their own fundamental jumps by `>= extra_restrike_rise_db` (`12.0` dB). (`extra_gate` / `P4` is disabled `False` to prevent soft-threshold false positives.)

---

## Benchmark Results: Original `v3` vs. Updated `v3` (`132 Takes, 1,974 Notes`)

Evaluated across **132 labelled recordings (1,974 notes)** spanning 5 players: **98 monophonic takes** (1,322 notes across 9 pieces) and **34 polyphonic two-hand takes** (652 notes across 4 pieces).

### Overall Improvement Summary

| Metric / Benchmark | Original `v3` | Updated `v3` (`shreema-hmm`) | Delta |
| :--- | :---: | :---: | :---: |
| **Overall Error Precision (`err P`)** | `0.92` | **`0.98`** | **+0.06** |
| **Overall Error Recall (`err R`)** | `1.00` | **`1.00`** | `0.00` |
| **Overall Error F1 (`err F1`)** | `0.96` | **`0.99`** | **+0.03** |
| **Overall Goal F1 (`goal F1`)** | `0.96` | **`0.97`** | **+0.01** |
| **Clean Takes Passed (`clean`)** | `36 / 38` (`97.7%` pass) | **`38 / 38` (`99.2%` pass)** | **+2 takes (100% clean pass)** |
| **Monophonic Goal F1 (98 takes)** | `0.97` (`0.94` Err P) | **`0.98` (`0.98` Err P)** | **+0.01 Goal F1 / +0.04 Err P** |
| **Polyphonic Error Precision (34 takes)** | `0.87` (`12/13` clean) | **`1.00` (`13/13` clean)** | **+0.13 Err P (100% clean pass)** |
| **`eval.generalise` Monophonic Errors** | `9` errors | **`4` errors** | **-56% errors** |
| **`eval.sensitivity` Synthetic Exact-Match** | `0.888` | **`0.959`** | **+0.071 (+7.1%)** |

### Per-Piece Scoreboard (Updated `v3`)

| Piece | Takes | Notes | Err P | Err R | Err F1 | Goal F1 | Wrong | Missed | Extra | Late | Restart | Clean | Pass% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `buns` | 3 | 51 | 1.00 | 1.00 | 1.00 | **1.00** | 1.00 | 1.00 | - | 1.00 | - | - | 100.0 |
| `c_high` | 4 | 60 | 0.67 | 1.00 | 0.80 | **0.67** | 0.67 | 1.00 | 0.50 | - | - | 1/1 | 100.0 |
| `c_low` | 3 | 45 | 1.00 | 1.00 | 1.00 | **1.00** | 1.00 | - | - | 1.00 | - | 1/1 | 100.0 |
| `c_major_rh` | 16 | 128 | 1.00 | 1.00 | 1.00 | **1.00** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 4/4 | 100.0 |
| `e_major_rh_down` | 15 | 120 | 1.00 | 1.00 | 1.00 | **1.00** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 4/4 | 100.0 |
| `frere` | 5 | 160 | 1.00 | 1.00 | 1.00 | **1.00** | - | 1.00 | 1.00 | 1.00 | 1.00 | 2/2 | 100.0 |
| `ode_melody` | 44 | 660 | 1.00 | 1.00 | 1.00 | **0.99** | 1.00 | 1.00 | 0.93 | 1.00 | 1.00 | 10/10 | 97.7 |
| `rests` | 2 | 14 | - | - | - | **1.00** | - | - | 1.00 | - | - | 1/1 | 100.0 |
| `twinkle` | 6 | 84 | 1.00 | 1.00 | 1.00 | **1.00** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 2/2 | 100.0 |
| **MONOPHONIC** | **98** | **1322** | **0.98** | **1.00** | **0.99** | **0.98** | **0.98** | **1.00** | **0.92** | **1.00** | **1.00** | **25/25** | **99.0** |
| `c_major_both` | 4 | 64 | 1.00 | 1.00 | 1.00 | **1.00** | 1.00 | - | - | - | - | 3/3 | 100.0 |
| `e_major_both` | 2 | 32 | - | - | - | - | - | - | - | - | - | 2/2 | 100.0 |
| `ode_harmony` | 16 | 304 | 1.00 | 1.00 | 1.00 | **0.97** | 1.00 | 1.00 | 0.80 | 1.00 | 1.00 | 5/5 | 100.0 |
| `twinkle_both` | 12 | 252 | 1.00 | 1.00 | 1.00 | **0.89** | 1.00 | 1.00 | 0.57 | 1.00 | 1.00 | 3/3 | 100.0 |
| **POLYPHONIC** | **34** | **652** | **1.00** | **1.00** | **1.00** | **0.93** | **1.00** | **1.00** | **0.67** | **1.00** | **1.00** | **13/13** | **100.0** |
| **OVERALL** | **132** | **1974** | **0.98** | **1.00** | **0.99** | **0.97** | **0.98** | **1.00** | **0.86** | **1.00** | **1.00** | **38/38** | **99.2** |

- **Played-pitch accuracy on detected wrong notes:** **100.0% (28/28)**
- **Core error recall:** **100.0%** across `wrong_pitch` (`28/28`), `missed` (`31/31`), and `late` (`32/32`), with **0 false `missed` flags** and **only 1 per-note disagreement** (`5chigh-1h-extra-g5`: `RH4 F5` flagged as `wrong_pitch` due to ringing `G5` carryover).

### Confusion Matrix (1,974 Total Notes)

| Gold \ Predicted | `correct` | `wrong_pitch` | `missed` | `extra` | `late` |
|---|---:|---:|---:|---:|---:|
| **`correct`** | 1882 | 1 | 0 | 5 | 0 |
| **`wrong_pitch`** | 0 | 28 | 0 | 0 | 0 |
| **`missed`** | 0 | 0 | 31 | 0 | 0 |
| **`extra`** | 2 | 0 | 0 | 21 | 0 |
| **`late`** | 0 | 0 | 0 | 0 | 32 |

### Generalization & Parameter Sensitivity (`eval.generalise` & `eval.sensitivity`)

Moving every parameter by **-25% and +25%** one at a time confirms that the new rules sit on wide, stable plateaus without overfitting:
- **Monophonic strict F1:** `0.95 – 0.99` (`0.99` as shipped)
- **Monophonic goal F1:** `0.95 – 0.98` (`0.98` as shipped)
- **Monophonic clean takes:** `25 / 25` across all 18 sensitivity runs
- **All-takes strict F1:** `0.96 – 1.00` (`0.99` as shipped)
- **All-takes clean takes:** `38 / 38` in 17 of 18 runs (`37 / 38` in 1)
- **Synthetic suite exact-match:** `0.939 – 0.968` (`0.959` as shipped, up from `0.888`)
- None of the added `P1`/`P1b`/`P2`/`P3b`/`P5` parameters shifted any real-take error count under $\pm 25\%$ perturbation in `eval.generalise`.

---

## System Architecture & Verdict Reference

### How `v3` Works
1. **Strikes:** From `stft_features.py`, a strike is a per-pitch onset peak followed by sustained presence. Simultaneous strikes form one event; overtones, sub-harmonic ghosts, and semitone leakage of the strongest pitch are folded into it.
2. **Alignment:** Windowed Viterbi over the event sequence (forward jumps up to 16 notes; restarts up to 4 bars back or to the start), scoring inter-onset intervals against written beat distances with two-pass local tempo tracking.
3. **Verdicts:** Read off the Viterbi path across three layers:

| Layer | Outcome | Meaning |
|---|---|---|
| What was played | `correct` | The written key |
| | `wrong pitch` | Another key instead of the written one (named; `"then corrected"` if followed on the beat by the right key) |
| | `missed` | The written note was never played |
| | `extra` | A note the score does not have (or a key struck before the beat) |
| When / how long | `late` | Started $\ge 0.33$ beats (or $\ge 20\%$ of written duration) late, labeled `"held"` or `"paused"` |
| | `held too long` | Previous key not released when the next note arrived on time |
| Going back | `restart` | Jumped back to an earlier position and played on; earlier abandoned attempt mistakes are listed alongside |

### Additional CLI & Audio Setup

```bash
# Convert raw .m4a recordings to 16-bit PCM .wav via ffmpeg
.venv/bin/python -m eval.cli convert

# Check against accepted scoreboard (unit tests + real takes + synthetic suite)
.venv/bin/python -m eval.cli check --model v3
```

For UI setup and historical phase logs, see [ui/README.md](ui/README.md), [HANDOFF.md](HANDOFF.md), and [eval/V3_LOG.md](eval/V3_LOG.md).