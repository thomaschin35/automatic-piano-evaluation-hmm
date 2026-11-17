# Handoff: where the v3 evaluator stands and what to do next

Written 2026-11-17 for whoever continues this project (a person or a model). Branch
`thomas-hmm-2`. Read this first, then [README.md](README.md) for how the model works and
[eval/README.md](eval/README.md) for adding recordings.

## 1. What exists

- **Model:** `stft_hmm_v3.py`. A note-level HMM: strike events are detected from the
  `stft_features.py` front end, a Viterbi aligns events to score positions (next, skip = missed,
  extra, back = restart), and verdicts are read off the path. `V3Config` holds every setting with a
  comment saying what it means.
- **Reported per note:** correct / wrong pitch (key named) / missed, plus extra notes, a `late`
  flag (with reason "held" or "paused"), restarts, and mistakes made before a restart.
  `held_long` (key not released under the next note) is defined in `eval/contract.py`.
- **Evaluation harness:** `eval/`. Labels per take in `eval/labels/`, the take list in
  `eval/manifest.toml`, the shared label sheet `eval/sheets/labels_manual_errors.csv` (mixed CRLF/LF
  line endings: edit it line by line, never rewrite it with the csv module).

## 2. Data

| Set | Takes | File prefix | Notes |
|---|---|---|---|
| Players 1-2 (interviews) | 16 one-hand, 4 two-hand, plus 5 unprefixed early takes | `1`, `2` | the first tuning set |
| Player 3 | 27 one-hand, 6 two-hand | `3` | shot list [eval/SHOT_LIST.md](eval/SHOT_LIST.md) |
| Round 2 (player 4) | 46 one-hand | `4` | shot list [eval/SHOT_LIST_2.md](eval/SHOT_LIST_2.md); was held out, then looked at on 2026-11-13, so it is no longer a blind test |
| Round 3 | **not recorded** | `5` | six new pieces, clean takes only, [eval/SHOT_LIST_3.md](eval/SHOT_LIST_3.md) |

91 labelled one-hand takes and 13 two-hand takes are scored. Canon and the three overlap takes are
excluded on purpose (the `exclude` line in the manifest says why). Same piano and phone throughout.

## 3. Scores today (2026-11-17, commit on `thomas-hmm-2`)

One-hand, 91 takes (found / false / not found):

| Class | Result |
|---|---|
| wrong pitch | 23 / 0 / 0 |
| missed | 22 / 0 / 0 |
| extra | 13 / 0 / 1 |
| late | 18 / 0 / 0 |
| clean takes with no flag | 30 / 30 |
| **goal F1** | **0.993** |

How it got there: 0.87 (2026-11-13) -> 0.93 (strike strength relative to each key's level in the
recording) -> 0.99 (second key struck with a note is an extra; stray key far ahead of a note is an
extra; re-strike accepted at the recording's loud level; late from 1/3 beat; two labels corrected
from the audio). Each step is in `eval/V3_LOG.md` with the full scoreboard.

**These numbers are measured on takes that were looked at while fixing.** They say "no known
failure on this piano, this phone, these three tunes", not "99% on new music". See section 5.

Two-hand and held-too-long: see section 6.

## 4. How to check a change

```bash
.venv/bin/python -m unittest discover eval/tests          # 57+ tests, seconds
.venv/bin/python -m eval.cli check --model v3             # real takes + synthetic suite, diff vs accepted
.venv/bin/python -m eval.cli check --model v3 --accept --log "what changed"
.venv/bin/python -m eval.generalise                       # every setting at +/-25%, by piece and player
.venv/bin/python -m eval.cli run --model v3               # per-take charts and per-note CSV
```

Rules that kept the work honest:

1. Measure the feature values behind a failure before changing anything; most "model" failures
   were a threshold sitting just under how a register measures, or a wrong label.
2. A new rule must follow from how a piano sounds, not from one take. Move each new number by
   +/-25% (`eval.generalise`) and keep it only if the result holds.
3. Never accept a scoreboard that worsens real one-hand results; check the synthetic suite too.
4. When a label and the audio disagree, check the strike count and timing in the features before
   changing the label, and say so in the sheet's comment column.

## 5. How general is it

`python -m eval.generalise` moves each of the 45 numeric settings by +/-25% and counts wrong
verdicts per piece and per player. The fragility is concentrated in three takes, not spread over
pieces or players:

| Take | Flips under | Why it is hard |
|---|---|---|
| `4cmajor-1h-wrong-octave1` | 8 of 14 sensitive settings | C5 played for C4: the wrong key is the right key's octave, so it reads as an overtone; C5 measures about 0.38 |
| `4cmajor-1h-extra-fsharp` | strike-level settings | the F#4 brushed with G4 measures 0.54, near the extra cut-off |
| `3ode-1h-shift` | skip / extra cost | hand one key off for three notes: reading it as three wrong notes or as skip + extras is genuinely close |

Everything else holds at +/-25%. What has **not** been tested at all: another piano or phone, low
(below C4) or high (above E5) keys, eighth notes, rests, pieces longer than 15 notes, tempo
changes. Round 3 exists to test exactly these.

## 6. Open work, in priority order

### 6.1 Record round 3 (blocks the only real generalisation test)
16 clean takes, about 10 minutes, no labelling needed (anything flagged is a false alarm). The
piece files, intake rules and `recordings/round3/` already exist; record, then
`python -m eval.cli evaluate`. A few takes on a different piano or phone (prefix `6`) would be the
most valuable recordings in the project. A no-audio check already passes: a perfectly heard
performance of each of the six pieces raises no flags.

### 6.2 Graded strike evidence (main design change left)
Today a strike either counts or does not, at fixed cut-offs (`match_min_salience` 0.30,
`sub_min_salience` 0.40, `extra_report_salience` 0.40) applied after each key is scaled to its own
level. Each borderline case has needed its own rule (soft re-strike, weak substitute, restrike
ratio, second key). The plan:

1. Replace the cut-offs with one probability that an event is a real strike of key k, from three
   inputs already computed: scaled salience, whether the attack was full (`Event.full`), and the
   key's own level rise in dB. Fit it as a logistic regression on the 1,300 labelled notes
   (positives: matched correct notes; negatives: unmatched events in clean takes).
2. Use -log of that probability as the emission and extra costs, so the alignment weighs evidence
   instead of switching at a threshold.
3. Fold in `restrike_*`, `sub_weak_penalty` and `restrike_report_ratio`, then delete them.
4. Validate leave-one-piece-out and leave-one-player-out (fit on the others, score the held-out
   group) and require no loss against section 3.

It was not done on 2026-11-17 because the current rules score 0.99 on all existing takes and the
change can only be shown to help on new recordings; do it together with 6.1.

### 6.3 Two-hand
### 6.4 Held too long

(Sections 6.3 and 6.4 are filled in below once the two parallel investigations report.)

### 6.5 Smaller items
- The two remaining hard cases above (octave wrong note, quick double strike `4ode-1h-extra-d15-quick`:
  the second D4 is only a 2 dB bump on a ringing key, below the 3 dB re-strike rise).
- The 1/3-beat late threshold costs 4 false lates in about 3,000 synthetic cases (after a
  synthetic extra or restart); a reason to use graded timing evidence too.
- `README.md` "Sensitivity" table predates all of this; `eval.generalise` replaces it.
- Long pieces: alignment is windowed (a 480-note piece aligns in about 3 s), but each strike still
  scans every position; about 50 s at 1,900 notes.
