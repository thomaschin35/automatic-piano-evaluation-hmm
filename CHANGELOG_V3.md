# Change log: from `stft_hmm_v2` to v3 (branch `thomas-hmm-2`)

Every change made since the v2 model, in the order it was made, with what changed, why it was
changed, and what it did to the scores. Written 2026-11-17.

- **Starting point:** commit `a683318` ("adding README.md") on `stft-hmm-tempo-calibration`, which
  holds `stft_hmm_v2.py` (pitch-gated IOI tempo calibration, commit `2d1c2a1`).
- **End point:** commit `75f75e9` on `thomas-hmm-2`. 34 commits (30 without merges), nothing pushed.
- `stft_hmm.py` and `stft_hmm_v2.py` were never edited; v2 stays as the baseline.
- Every scoreboard quoted here is in full in [eval/V3_LOG.md](eval/V3_LOG.md). Current state and
  open work are in [HANDOFF.md](HANDOFF.md).

**How to read the scores.** "Goal F1" is the per-note F1 over the error classes the project cares
about, on one-hand ("mono") real takes. "Found / false / not found" counts are true positives / false
positives / false negatives. Synthetic cases are errors made by editing clean recordings (section 1).

---

## Score history at a glance

| Date | Commit | Change | One-hand takes | One-hand goal F1 | Two-hand goal F1 |
|---|---|---|---|---|---|
| 11-10 | `2a02f63` | v2 measured on the new scoreboard | 22 | 0.80 | 0.53 |
| 11-10 | `4763d94` | v3: HMM over key strikes | 22 | 0.93 | 0.59 |
| 11-10 | `d5946eb` | late / held verdicts | 22 | 0.93 | 0.50 |
| 11-10 | `4f39e49` | backtracking | 22 | 0.93 | 0.50 |
| 11-10 | `1d3b913` | two-voice strikes | 22 | 0.93 | 0.53 |
| 11-12 | `f5b405e` | player 3 added (not tuned on) | 52 | 0.85 | 0.48 |
| 11-13 | `2dc5ab8` | taxonomy, test-set cleanup, soft re-strikes | 45 | 1.00 | 0.56 |
| 11-13 | `43d19d8` | round 2 added (held out) | 91 | 0.87 (round 2: 0.77) | 0.56 |
| 11-13 | `2d8d536` | strike level relative to each key | 91 | 0.93 (round 2: 0.86) | 0.56 |
| 11-17 | `5b9191e` | one-hand misses fixed | 91 | **0.99** | 0.56 |
| 11-17 | `e142941` | two-hand fixes | 91 | 0.99 | **1.00** |
| 11-17 | `75f75e9` | held-too-long detector | 91 | 0.99 | 1.00 |

The take count changes because recordings were added (player 3, round 2) and because Canon and the
three overlap takes were removed on 11-13 (step 12). The definition of "goal F1" also widened on
11-13 to cover all five classes, so numbers before and after step 12 are not strictly comparable.

---

## Why v3 exists: what the v2 review found

Measured before any change (scoreboard "Phase 0" in `eval/V3_LOG.md`):

- One-hand "any error" F1 was 0.88, but on only 28 labelled error notes, and it hid label mistakes.
- **v2 was steered by pitch, not by timing.**
  - Its IOI tempo curve, `min_dur_fraction` and `max_dur_fraction` changed no prediction.
  - With the pitch term switched off, precision fell to 0.17.
- **Grading only asked "is the expected pitch present?"**
  - A missed note was invisible when the next note had the same pitch: 15% found in the wrong-score test.
  - Missed notes were often labelled wrong pitch.
  - Extra notes were never reported (82% silent).
  - There were no timing verdicts at all.
- **The state graph could not express common mistakes.** It had no skip, extra, pause or end state
  and no real duration model, so a skipped, added or repeated note could not be represented.
- **Two-hand false alarms** were almost all right-hand notes sitting on an octave, fifth or third
  harmonic of the left hand, which the front end's overtone subtraction erased.

The plan (`~/.claude/plans/please-create-a-plan-synchronous-crane.md`) was:
- a new model file, `stft_hmm_v3.py`, with v2 kept as the baseline;
- one measuring command run after every change;
- no new recordings needed, because errors would be synthesised from clean takes.

---

## 1. Measuring stick (`2a02f63`, 2026-11-10)

**What changed**
- `eval/check.py` and `python -m eval.cli check`: one command prints a full scoreboard. It covers
  the unit tests, every real take, a synthetic error suite, an onset-only test, and a diff against
  the last accepted scoreboard.
- `eval/synth.py`: makes labelled errors from clean takes in two ways.
  - Wrong-score cases: change the score instead of the audio.
    - `sub`: a wrong pitch.
    - `shift`: the hand one key off.
    - `miss_ins_step` / `miss_ins_repeat`: a missed note, including a missed note inside a repeat.
    - `miss_del`: a missed note removed from the score.
    - `extra_score` / `extra_dup`: an extra note.
  - Spliced-feature cases: cut and repeat the audio features.
    - `hesitate`, `stop_early`, `restart`.
- `eval/cache.py`: saves each take's front-end features to disk, so the synthetic suite runs in
  seconds instead of minutes.
- `eval/metrics.py`: strict per-class precision, recall and F1, plus a one-hand-only block and a
  clean-take pass count.
- `eval/scoreboards/v2.json`: v2's frozen scoreboard.
- Tests in `eval/tests/test_synth.py`.

**Why**
- The review showed the existing "any error" score was too coarse. It could not show which kind of
  mistake failed, and 28 error notes were too few to tune on.
- Synthetic errors gave hundreds of labelled cases per kind without asking anyone to record.

**Result**
- v2 reproduced the review's numbers: one-hand goal F1 0.80 and two-hand 0.53.
- Missed notes in a repeat were 7% exact.
- Extras were never found (0.00).
- Restarts were never located (0.00).

## 2. v3: a note-level HMM over detected key strikes (`4763d94`, 2026-11-10)

**What changed**
- New file `stft_hmm_v3.py`, registered as adapter `"v3"` in `eval/adapters.py`.
- **Strikes are detected first.** A strike is a per-key onset peak followed by the key's sound
  staying present.
- **The HMM runs over the strike sequence**, not over frames. The hidden state is the score
  position, and each strike can be:
  - the next note;
  - a wrong pitch for it;
  - a skip (missed notes);
  - an extra.
- **Timing counts.** The time between strikes (IOI) is compared with what the score and the local
  tempo predict.
- **Strike salience** = sustained presence after the onset × onset strength.
  - On the labelled takes, every real strike had an onset score ≥ 0.65 and salience ≥ 0.35.
  - Noise sat below both, so these became the thresholds.
- Two conventions were set to match how the hand labels were written:
  - **A wrong key followed by the right key** is reported as a wrong pitch on that note, marked as
    corrected. Only a re-strike of a neighbouring note is an extra.
  - **A missed note inside a run of equal pitches** (E E E played as E E) cannot be placed by ear,
    so a "missed" anywhere in that run is accepted.

**Why**
- v2's frame-level chain had no way to say "this note was skipped" or "this strike was extra".
- Decoding strikes makes missed, extra and repeated notes into ordinary states of the model.
- Phases 1-3 of the plan (grading, onset stream, state graph) became one model, checked against
  all three of their gates.

**Result**
- One-hand strict F1 0.85 → 0.98; goal F1 0.80 → 0.93.
- 12/12 clean one-hand takes passed.
- Synthetic missed notes in a repeat 0.07 → 0.94 exact; extras found 1.00.

## 3. Timing allowances for skipped and added notes; smudge rule (`b64695d`, 2026-11-10)

**What changed**
- **Skipped notes may drop their time** (`skip_time_slack` 1.0): a skipped note's time may be
  dropped along with it, rather than always kept.
- **Added notes may add their time** (`extra_time_slack` 1.5): an extra note may add time of its own.
- **Wrong neighbouring key is cheaper** (`sub_cost_near` 2.5): a wrong key within a whole tone of
  the right one costs less than a far key, because it is the usual slip.
- **Smudge rule** (`smudge_ratio` 0.90): a stray key struck at least 90% as strongly as the right
  key, at the same moment, means the note was not clean.
- **Salience cut-offs lowered:**
  - `sub_min_salience` 0.45 → 0.40;
  - `extra_weak_salience` 0.35 → 0.30;
  - `extra_strong_salience` 0.60 → 0.45;
  - `extra_report_salience` 0.50 → 0.40.
- Extra-note matching in `eval/metrics.py` was made tolerant of runs of the same pitch.

**Why**
- In the synthetic missed and extra cases, the alignment made a timing error to avoid a skip or an
  extra. A player who drops a note usually also drops its beat, and the model could not represent that.
- The salience cuts were too strict for soft extras.

## 4. Late and held-too-long verdicts (`d5946eb`, 2026-11-10)

**What changed**
- **A timing error** is a gap between two notes of ≥ 1.5× its written length at the local tempo.
  - If the first note was still sounding up to the next strike, it was *held too long*.
  - If it had died away, the next note was *late*.
- New synthetic kinds `hold` and `gap` (`eval/cache.py` stretches a note's features).

**Why**
- v2 had no timing verdicts.
- The 1.5× threshold (half a note value over) was set by principle, not fitted. The 8 timing
  labels were not consistent enough to fit: some labelled gaps measured 1.0-1.1×, and an
  unlabelled one measured 2.4×.

**Result**
- One-hand late: 4 found / 1 false / 1 not found.
- Synthetic `hold` 0.94 and `gap` 1.00 found.
- Accepted with `--force`, because two-hand strict F1 fell 0.78 → 0.76. The changed salience
  thresholds lost right-hand notes sitting on left-hand harmonics, which was step 6's job.

*This definition of "held too long" was replaced in step 12 and the detector rebuilt in step 21.*

## 5. Backtracking (`4f39e49`, 2026-11-10)

**What changed**
- **Backward arcs:** the path may return from any note to an earlier one. The prior prefers, in
  order: the same note, a bar start, a phrase start, then the piece start.
- The report says where the player went back from and to.

**Why**
- Backtracking was one of the two project goals.
- Without backward arcs, a restart was read as a long run of wrong and extra notes: 5.8 false
  flags per synthetic restart.

**Result**
- Synthetic restarts located 0.00 → 1.00, with 0 false flags; everything else unchanged.

## 6. Two voices: strikes read off the raw spectrum (`1d3b913`, 2026-11-10)

**What changed**
- **Raw-spectrum strikes.** Some score pitches sit on a harmonic of another score pitch. For
  those, a jump of the fundamental in the raw spectrum counts as a strike when the score expects
  that note there. The jump must be ≥ 6 dB and stand ≥ 10 dB above its neighbours.
  - 6 dB rather than 10 dB, so that a soft re-strike of a still-ringing note over a held bass is
    seen (`odeharm` RH10).
- **Crashes are now caught.** `check` reports crashed predictions and refuses to accept a
  scoreboard that has any.
- First v3 unit tests (`eval/tests/test_v3.py`) on a synthetic performance.

**Why**
- The front end's overtone subtraction removed right-hand notes on left-hand harmonics, for
  example octave scales and the Ode melody's C4/G4 over C3/G3.
- I measured three cues for telling a doubled octave from a single note: even-vs-odd partial
  balance, h2−h1, and h2−h3. They overlap too much to decide per note, so v3 does not try.
- A bug in this step made predictions crash, and the old scoreboard counted a crash as "all
  correct", so it showed 17/17 clean. That is why `check` now refuses crashed predictions.

**Known limit, by design**
- Sometimes the left hand strikes a note while the score expects its octave, fifth or third in the
  right hand at the same moment. If that right-hand note is *missing*, it cannot be told from the
  bass's overtone, so it is reported as correct.
- A *wrong* right-hand note there is still caught.
- False alarms on correct playing were judged worse than this blind spot.

**Result**
- Clean takes 13/17 → 17/17.
- Two-hand strict F1 0.54 → 0.60; one-hand unchanged.

## 7. Documentation, like-for-like v2 baseline, sensitivity sweep (`1cdf859`, 2026-11-10)

**What changed**
- `README.md` gained v3 results, limits and a sensitivity table.
- `eval/scoreboards/v2.json` was re-run with exactly the same takes and metric code as v3.
- New `eval/sensitivity.py` moves each setting and re-runs the synthetic suite.

**Why**
- The v2 baseline had been made before some metric changes, so the comparison was not like-for-like.
- The sweep shows whether a threshold is sitting on a cliff edge.

## 8. Shot list for a held-out player (`fa9d5b9`, `cbfefcf`, 2026-11-10)

**What changed**
- [eval/SHOT_LIST.md](eval/SHOT_LIST.md): an exact list of takes for a new player (player 3) to
  record, balanced to 5 takes per error type.
- `check` skipped a "test" split.

**Why**
- Every threshold so far had been read off players 1-2. A player the model had never seen was the
  only honest test of whether those thresholds generalise.

## 9. Intake of player 3; dev/test split dropped (`8f49d53`, `12c4b98`, 2026-11-12)

**What changed**
- `eval/intake.toml` scans `recordings/person3`.
- The 36 player-3 takes and 3 older two-hand takes were registered in `eval/manifest.toml`.
- The dev/test split was removed: every take is scored, player 3 in its own rows, and v3 is the
  default model.

**Why**
- Reporting player 3 in its own rows gives the same held-out view without maintaining two code
  paths, and lets every later change be checked on all takes.

## 10. Player 3 labelled; new baseline (`f5b405e`, 2026-11-12)

**What changed**
- 36 label files under `eval/labels/` were written from the shot list (each take was recorded to
  a known script).
- `eval/sheets/labels_manual_errors.csv` gained the matching rows.
- The scoreboard was accepted with `--force` as the new baseline. Metrics "regressed" only because
  the take set grew.

**Result: the first held-out test, and the model did not hold up**
- One-hand goal F1 fell from 0.93 to 0.85 over 52 takes.
- On player 3 alone, goal F1 was 0.73 and strict precision 0.72.
- The failures were:
  - soft re-strikes read as missed;
  - extras read as wrong pitches;
  - held-too-long confused with late;
  - two-hand left-hand notes missed.

## 11. Per-take alignment charts and a full confusion matrix (`7e296e8`, 2026-11-12)

**What changed**
- `eval/plots.py` draws per-take v3 alignment charts: strikes, the score path, and the verdicts.
- `eval/metrics.py` gives a confusion matrix over all classes, including extra, held and late.

**Why**
- The player-3 failures needed to be looked at one by one, to tell a model failure from a label
  problem.

## 12. Two-layer taxonomy, test-set cleanup, soft re-strikes, late in beats (`2dc5ab8`, 2026-11-13)

**Taxonomy (`eval/contract.py`, `eval/metrics.py`)**
- **What changed:** verdicts became two layers.
  - *What was played:* correct, wrong pitch, missed, extra.
  - *When it was played:* **late** means the note started late for any reason. **held_long**
    means the key was not released under the next note (an overlap).
  - held_long was defined but no longer detected at this point.
  - Goal F1 now covers all classes.
- **Why:** "held" and "late" meant different things in different labels. A long note before a
  pause was sometimes labelled held, sometimes late. Defining late by the note's start, and held
  by overlap, makes the labels and the model agree.

**Test set**
- **What changed:**
  - Canon and the three overlap takes were excluded. The reason is written on the `exclude` line
    in the manifest.
  - The stretch takes were relabelled late.
  - In `3ode-1h-pause4`, RH4 was relabelled missed.
- **Why:**
  - Canon's timing labels were made by ear before the late / held definitions were fixed, so
    they could not be scored against the new definitions.
  - The overlap takes were about held notes, which were out of scope until step 21.
  - The stretch and pause4 labels contradicted the audio and the new definitions.

**Model**
- **Soft re-strikes:** a ringing key struck again is accepted from a lower onset score, when its
  own level and the frame energy both rise (`restrike_*` settings).
  - Why: player 3's soft repeats of the same key were read as missed.
- **Soft neighbour repeats:** a soft repeat of a neighbouring key is reported as an extra.
- **Faint wrong key:** a faint wrong key can stand in for a note, at a higher cost (`sub_weak_penalty`).
  - Why: a soft wrong key was dropped as noise, and the note was then reported missed instead of
    wrong pitch.
- **Late in beats:** late is measured in beats, not as a ratio of the written length, and is also
  checked after a missed note.
  - Why: a ratio over-flags short notes and under-flags long ones, and a note after a missed one
    was never judged.

**Also:** [eval/SHOT_LIST_2.md](eval/SHOT_LIST_2.md) was written, a 46-take shot list for a new
player (round 2), to be held out.

**Result**
- One-hand goal F1 1.00 over 45 takes.
- This was tuned on those takes, which is why round 2 was to be held out.

## 13. Round 2: 46 held-out one-hand takes (`43d19d8`, 2026-11-13)

**What changed**
- 46 takes from player 4 were labelled from `SHOT_LIST_2.md` and added to the manifest and the sheet.

**Result: the second held-out test, and it fell again**
- Round 2 goal F1 was 0.77; all one-hand takes together scored 0.87.
- The failures pointed at one cause: soft high notes (D#5, C#5) were read as missed, and several
  soft extras were not found.
- This led directly to step 15.

## 14. Restarts became a labelled, scored class (`0cd0b24`, 2026-11-13)

**What changed**
- A sheet comment `restart X -> Y` is parsed into the label (`eval/sheet.py`, `eval/labels.py`).
- Restarts are scored as their own class.
- 8 restart takes were labelled.

**Why**
- Backtracking was a goal, but on real takes it was only checked indirectly, through "no false
  flags". Now the model must name the right restart point.

**Result:** restarts 8 found / 0 false / 0 not found.

## 15. Strike strength relative to each key's level; no strikes in silence (`2d8d536`, 2026-11-13)

**What changed**
- **Each key gets its own level** (`key_scales`).
  - Each key's level is read from this recording's own clear strikes.
  - A key seen once or never borrows its level from its neighbours.
  - A quiet key's strikes are scaled up to `level_reference` 0.65, the level the thresholds were
    written for. Scaling only raises a key (`level_scale_min` 1.0) and is capped at
    `level_scale_max` 2.5.
- **Silence check:** a strike candidate where the frame energy is below `strike_energy` 0.01 (the
  recording's quiet floor) is dropped.
- **Round-3 setup:**
  - six pieces v3 was never tuned on: `twinkle`, `buns`, `frere`, `c_low`, `c_high`, `rests`;
  - intake rules;
  - a clean-takes-only shot list, [eval/SHOT_LIST_3.md](eval/SHOT_LIST_3.md);
  - a no-audio test that a perfect performance of each piece raises no flags.
- Unit tests `RelativeStrikeLevels` and `NewPieceShapes`.

**Why**
- Salience was compared with fixed numbers (0.30 / 0.40), but keys differ a lot on one piano. C4
  reads about 0.76, while C5-E5 read about 0.30-0.45 even with full attacks.
- So soft high notes fell under the cut-off and were called missed. That was round 2's main failure.
- The scaling only raises quiet keys, so it cannot hide a real soft mistake on a loud key.
- The silence check stops scaled-up noise from becoming a strike.

**Result**
- One-hand goal F1 0.872 → 0.926; clean takes 29/30 → 30/30.
- Round 2: 0.77 → 0.86. Players 1-3 stayed at 1.00, and two-hand was unchanged.
- The result held at 0.920-0.926 across 13 variants of the new settings.

## 16. Backtracking: mistakes before a restart kept; windowed search (`ec6ffdf`, 2026-11-13)

**What changed**
- **Attempts.** Each restart starts a new attempt.
  - Every note is judged by the last attempt that reached it.
  - Mistakes made in abandoned attempts are kept (`Report.earlier`, `TakePrediction.earlier`) and
    printed with the restart, for example: "WENT BACK from note 7 to note 5, after: note 7 WRONG
    NOTE (played D#4)".
  - Late is judged per visit.
- **Labels and synthetic cases.**
  - A sheet comment such as `attempt 1: RH7 wrong_pitch played D#4` becomes an `[[earlier]]` label,
    scored as its own class.
  - New synthetic kind `restart_wrong`: a wrong key on the last note before going back.
  - Takes with a labelled restart are no longer used as clean sources for synthetic cases.
- **Faster alignment.** `align()` searches only moves a player makes: a forward jump of at most 16
  notes, or a restart at most 4 bars back or to the beginning. Emissions are computed once per
  pitch set. `align_dense()` is kept as the reference for tests.

**Why**
- Before, the mistake that caused a restart (usually the reason for it) was overwritten by the
  later, correct attempt. A teacher wants to see it.
- A slow gap in an abandoned attempt wrongly flagged the note's final verdict as late.
- The exhaustive search took about 6 minutes on a 480-note piece.

**Result**
- Identical to the exhaustive search on all real takes (312 checks) and on a long piece.
- 480 notes: about 6 min → about 3 s.
- Synthetic `restart_wrong`: located 1.00, mistake before the restart found 1.00.

## 17. Scoreboard accepted; housekeeping (`d10b6ed`, 2026-11-13)

- The backtracking scoreboard was accepted.
- The stale `eval/handoff_fast_playing.md` was removed.
- `eval/label_sheet.tsv` was moved to `eval/sheets/label_sheet_round1.tsv`.

## 18. Visuals and evaluation session work (`f451149`, 2026-11-13)

**What changed**
- Tempo charts (`eval/tempo.py`).
- Plot updates (`eval/plots.py`).
- Run storage (`eval/runs.py`).
- Metric breakdowns (`eval/metrics.py`).
- Piece helpers (`eval/pieces.py`).
- The results notebook (`eval/results.ipynb`).

**Why:** to look at results per take and per tempo while reviewing. No model change.

## 19. One-hand misses fixed from the measurements; two labels corrected (`5b9191e`, 2026-11-17)

Each of the remaining one-hand failures was measured in the features before anything was changed.

**Second key struck together with a correct note is an extra**
- **Rule:** the key must meet all of these:
  - it has its own full attack;
  - it is as clear as any reported extra (≥ `extra_report_salience`);
  - it is not one of the previous note's keys (those are still ringing);
  - it is not at an overtone interval from the note.
- **Why:** a brushed neighbouring key (F#4 with G4, A4 with G4) was silently absorbed into the
  correct note.
- The overtone and previous-key guards were added after the first version caused false extras on
  two-hand takes (G2 with D4) and in synthetic `sub`/`shift` cases.
- **Found:** the brushed F#4 and A4.

**Stray key far ahead of a note is an extra**
- **Rule:** a stray key more than `correction_max_beats` 1.5 beats before the note it would be
  read as "corrected into" is an extra, not a wrong-then-corrected note.
- **Why:** an extra played before the first note was reported as a wrong pitch on note 1. A real
  correction follows within a beat or so.

**Re-strike at the recording's loud level**
- **Rule:** if the energy before the strike is already near the top (above 1 − `restrike_energy`),
  the energy rise is not required.
- **Why:** the energy meter saturates at the recording's loud level, so a loud re-strike of a
  ringing key cannot show a rise and was read as missed.

**Late from 1/3 beat** (`late_beats` 0.40 → 0.33)
- **Why:**
  - The two labelled half-beat lates measured 0.35 and 0.39 beats.
  - Every note not labelled late measured ≤ 0.27 beats, apart from notes next to extras, which are
    not judged.
- **Cost:** 4 new false lates in about 3,000 synthetic cases.

**Labels corrected after checking the audio**
- `4ode-1h-wrong-anticip3` RH3 changed from wrong pitch to **missed**. The take has 14 strikes,
  only two of them G4s, and F4 was dropped, not replaced.
- `4cmajor-1h-wrong-octave1` RH2 changed to **late**. It arrives 2 beats after RH1 while the hand
  moves down.
- Both rows were edited in the sheet, with the reason in the comment column.

**Tests:** class `SecondKeyAndStrayKey`.

**Result**
- One-hand goal F1 0.926 → **0.993**:
  - wrong 23/0/0;
  - missed 22/0/0;
  - extra 13/0/1;
  - late 18/0/0;
  - clean 30/30.
- Two-hand unchanged. Every setting holds at ±25%.
- Still not found: the quick double D4 in `4ode-1h-extra-d15-quick`. The second D4 is only a 2 dB
  bump on a ringing key, below the 3 dB re-strike rise.

## 20. Two hands (`f1975b5`, `1eb4761`, accepted in `e142941`, 2026-11-17)

Done in a separate worktree, then merged (`be314e5`, `ce9734c`). There were six separate causes,
each fixed with a rule based on how two piano notes sound together:

| Failure | Cause found in the features | Rule now |
|---|---|---|
| Left-hand E3 under an octave missed | its onset score peaks at 0.43 beside the louder upper note; only the *upper* note of a written interval was read off the raw spectrum | the lower note of a written octave/fifth/third is also read off the spectrum, if the front end hears it sounding afterwards |
| F3 missed | its raw level took the loudest of 3 bins, and the neighbouring bin still held E3 ringing | where the next semitone is under 3 bins away (the Hann window's main lobe), only the centre bin is used |
| First note of the take missed | the "before" window repeated the first frame, so there was no rise | the time before the recording starts counts as quiet |
| Missed right-hand C4 read as played | the jump was C3's own 2nd partial (the same −4 dB ratio as C3 alone) | an upper note during a lower-key strike must beat the leanest strike of that key in this recording by `raw_rise_db` |
| The other hand's extra C3 read as a smudge (wrong pitch) | a stray key was always attributed to the nearest expected note | a stray key more than `hand_span` (12) semitones away was played by the other hand: an extra, attached to that hand's last note |
| C4 re-struck after a pause read as C5 | its fundamental rose only 2 dB, its octave 10 dB | a candidate with no plain onset, no gain in presence, and at an overtone interval above a ringing key is that key struck again |

**Tests:** class `TwoVoices`.

**Result**
- Two-hand goal F1 0.556 → **1.000**; clean two-hand takes 6/7 → 7/7.
- One-hand unchanged.
- The re-strike rule changes 2 events in all 104 takes.

**Caveats**
- This was diagnosed on the same 13 takes it is scored on, so the number is optimistic.
- The last rule has a small margin: the case measures 1.03 against a 1.0 limit, while real new
  notes measure 1.2 or more.

## 21. Held too long detected again, conservatively (`6efdcf5`, `b122959`, 2026-11-17)

Done in a separate worktree, then merged (`4157d28`, `18059df`).

**What changed**
- `held_evidence` / `held_flags` run after the timing check. They judge only consecutive single
  notes that are:
  - on different keys;
  - both correct;
  - with the next one on time;
  - with no restart or stray strike between them.
- **The measurement:** how far the previous key's fundamental falls in the `held_after_s` 0.25 s
  after the next strike.
  - A released key falls a median of 24-28 dB there.
  - The two clearly held notes recorded fall only 5-6 dB.
- **The flag:** a note is flagged when its fall is under `held_drop_share` 0.33 of this
  recording's usual fall and the key is still sounding.
- **`b122959`:** the fall is read at its lowest point within the wait (a 3-frame mean), not at the
  window's end. A held key only fades, so a key whose sound comes back was struck or rang again.
- New synthetic kind `overlap` (`eval/cache.sustain` holds the key's harmonics under the next
  note). The adapter emits `held_long`, and the printed report shows it.
- **Tests:** class `HeldTooLong`.

**Why**
- Held too long was part of the stated goal, but since step 12 it was only defined, not detected.
- Each recording is compared with its own usual fall, because the room, the piano and the phone
  change how fast a released note dies.

**Result**
- Real takes: 0 false flags in 511 judged pairs; all other scores unchanged.
- The three excluded overlap takes were not used for tuning:
  - it flags the labelled note in 2 of them;
  - the third (`3emajor-1h-down-overlap4`) is not flagged, which matches the remark that it was
    not really held.
- Synthetic `overlap`: 105 of 178 found, 0 false `held_long`.
- 2 new false flags in about 3,000 other synthetic cases, both on copies of one take whose D#5
  rings on its own.
- `held_long` is reported but **not yet counted in goal F1**, because no real held-note labels
  exist in the scored set.

## 22. Generalisation check (`8b80345`, 2026-11-17)

**What changed**
- New `python -m eval.generalise` moves each of the 45 numeric `V3Config` settings by −25% and
  +25%, one at a time, and scores the one-hand real takes by piece and by player.

**Why**
- After this many rules, it needed checking whether any setting was fitted to particular
  recordings.
- A setting that moves only one piece's results was probably chosen for it.

**Result**
- The fragility sits in three takes, not across pieces or players:
  - `4cmajor-1h-wrong-octave1`: C5 for C4, which reads as an overtone;
  - `4cmajor-1h-extra-fsharp`: the brushed F#4 measures 0.54, near the cut-off;
  - `3ode-1h-shift`: three wrong notes and skip + extras are genuinely close readings.
- Everything else holds at ±25%.

## 23. Documents (`c78acf6`, `63abe61`, `75f75e9`, 2026-11-17)

- [HANDOFF.md](HANDOFF.md) covers:
  - what exists and the data;
  - today's scores and how to check a change;
  - how general the scores are;
  - the open work in priority order: round 3 recordings, the graded strike-evidence plan, two-hand,
    held too long, and smaller items.
- `README.md` status was brought up to date and points to HANDOFF.md.
- `eval/V3_LOG.md` and `eval/scoreboards/v3.json` were updated at each acceptance.

---

## What did not change

- `stft_hmm.py`, `stft_hmm_v2.py` and the front end `stft_features.py` were not edited. The
  two-voice work in step 6 and step 20 lives in `stft_hmm_v3.py`, not in the front end as first planned.
- Recordings were never edited.
- Labels changed only in the places named above, each with its reason in the sheet:
  - steps 12 and 19;
  - the new player-3, round-2 and restart labels.

## Considered and not done

- **Per-note doubled-octave detection (step 6):** the cues overlap too much.
- **Graded strike evidence**, replacing the fixed salience cut-offs with one fitted probability.
  Planned in HANDOFF.md section 6.2. It is held until new recordings exist, because the current
  rules already score 0.99 on every take there is, so a gain could not be shown.
- **Round 3 recordings:** not made. The pieces, intake rules and shot list are ready.
