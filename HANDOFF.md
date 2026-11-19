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
  `held_long` (key not released under the next note) is defined in `eval/contract.py` and detected conservatively (section 6.4).
- **Evaluation harness:** `eval/`. Labels per take in `eval/labels/`, the take list in
  `eval/manifest.toml`, the shared label sheet `eval/sheets/labels_manual_errors.csv` (mixed CRLF/LF
  line endings: edit it line by line, never rewrite it with the csv module).

## 2. Data

| Set | Takes | File prefix | Notes |
|---|---|---|---|
| Players 1-2 (interviews) | 16 one-hand, 4 two-hand, plus 5 unprefixed early takes | `1`, `2` | the first tuning set |
| Player 3 | 27 one-hand, 6 two-hand | `3` | shot list [eval/SHOT_LIST.md](eval/SHOT_LIST.md) |
| Round 2 (player 4) | 46 one-hand | `4` | shot list [eval/SHOT_LIST_2.md](eval/SHOT_LIST_2.md); was held out, then looked at on 2026-11-13, so it is no longer a blind test |
| Round 3 | **not recorded** | `5` | blind test: 25 one-hand + 24 two-hand takes with every mistake type, old and new pieces, [eval/SHOT_LIST_3.md](eval/SHOT_LIST_3.md). Label from the shot list, score once, write the blind result down before looking at any take |

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

Two-hand, 13 takes: goal F1 **1.000**, 7 / 7 clean (section 6.3).

**Round 3 (blind, 2026-11-19, 49 takes):** goal F1 0.61 one hand / 0.66 two hands when scored
blind; 0.79 / 0.75 after the phantom-key and ringing-key rules (eval/V3_LOG.md). Most real
mistakes are found; the remaining loss is false alarms and naming (see V3_LOG for the list). Held too long: reported, conservative, no false flags on real takes (section 6.4).

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
49 takes (25 one-hand, 24 two-hand, plus 3 optional held-too-long), about 50 minutes; the 20
takes marked ★ give a first answer in about 20 minutes. Every mistake type in both groups, on the old
tunes and on new ones (Twinkle, Buns, Frere, low and high C, rests, two-hand Twinkle). Protocol:
label the takes from the shot list (plus `recordings/round3/notes.txt` for takes that went
differently), run `python -m eval.cli check --model v3` **once**, write `mono_round3` and
`two_hand_round3` into `eval/V3_LOG.md` as the blind result, and only then look at takes. A few
takes on a different piano or phone (prefix `6`) would be the most valuable recordings in the
project.

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
Two-hand goal F1 went from 0.556 to **1.000** on 2026-11-17; all 7 clean two-hand takes now pass, and
one-hand results are unchanged. **These are the same 13 takes the causes were diagnosed on**, so
the number is optimistic. The new rules are narrow (the re-strike rule changes 2 events in all
104 takes), and each follows from how two piano notes sound together:

| Failure | Cause | Rule now |
|---|---|---|
| left-hand E3 under an octave, missed | its onset score peaks at 0.43; only the upper note of a written interval was read off the raw spectrum | the lower note of a written octave/fifth/third is also read off the spectrum, if the front end hears it sounding afterwards |
| F3 missed | its raw level took the loudest of 3 bins, and the neighbour bin held E3 still ringing | where the next semitone is under 3 bins away (the Hann main lobe), only the centre bin is used |
| first note of the take missed | the "before" window repeated the first frame | before the start of the recording counts as quiet |
| missed C4 read as played | it was C3's own 2nd partial (same -4 dB ratio as C3 alone) | an upper note during a lower-key strike must beat the leanest strike of that key in this recording by `raw_rise_db` |
| other hand's extra C3 read as a smudge | | a stray key more than `hand_span` (12) semitones away is the other hand: an extra, anchored to that hand's last note |
| C4 re-struck after a pause read as C5 | its fundamental rose only 2 dB, its octave 10 dB | a candidate with no plain onset, no presence gain and an overtone interval above a ringing key is that key struck again |

Fragile margin: the last rule needs a presence tolerance above 1.0 (the case measures 1.03; real
new notes 1.2 or more). Canon (the hardest two-hand piece) is still excluded; adding it back with
re-checked labels is the real two-hand test.

### 6.4 Held too long
A conservative detector exists since 2026-11-17 (`held_evidence` / `held_flags` in
`stft_hmm_v3.py`). For two consecutive single notes on different keys, both played correctly and
the next one on time, it measures how far the previous key's fundamental falls in the 0.25 s after
the next strike (`held_after_s`). A released key falls a median of 24-28 dB there; the two clearly
held notes recorded fall 5-6 dB. A note is flagged when its fall is under a third
(`held_drop_share`) of this recording's usual fall and it is still sounding.

| Test | Result |
|---|---|
| real takes, 511 pairs judged (no gold held notes) | 0 false flags; all other scores unchanged |
| excluded overlap takes (not tuned on) | 2 of 3 flagged on the labelled note; the third (`3emajor-1h-down-overlap4`) is not, which matches the user's remark that it was not held |
| synthetic `overlap` kind, 178 cases | 105 found, 0 false `held_long` |
| other synthetic kinds | 2 new false flags in ~3,000, both on spliced copies of one take whose D#5 rings on its own |

Limits: next notes shorter than 0.25 s are never judged; keys that ring by themselves look partly
held; chords, same-key repeats and anything next to a wrong note, stray strike or restart are
skipped; with pedal or heavy room ring the whole recording is skipped. `held_long` is reported but
**not yet counted in the goal score** (`eval/metrics.py` counts only `late` as timing); add it
when real held-note labels exist. Recording ten or so deliberate overlaps (and re-including the
overlap takes after re-checking their labels by ear) is the next step.

### 6.5 Smaller items
- The two remaining hard cases above (octave wrong note, quick double strike `4ode-1h-extra-d15-quick`:
  the second D4 is only a 2 dB bump on a ringing key, below the 3 dB re-strike rise).
- The 1/3-beat late threshold costs 4 false lates in about 3,000 synthetic cases (after a
  synthetic extra or restart); a reason to use graded timing evidence too.
- `README.md` "Sensitivity" table predates all of this; `eval.generalise` replaces it.
- Long pieces: alignment is windowed (a 480-note piece aligns in about 3 s), but each strike still
  scans every position; about 50 s at 1,900 notes.
