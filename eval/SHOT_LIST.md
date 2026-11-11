# Recording shot list: held-out test set (player 3)

**Purpose:** check v3 on a pianist it has never seen. Every number so far comes from players 1 and 2,
and v3's settings were chosen while looking at their takes. These recordings go in the **test**
split and are never used to tune anything.

**What you need:** a new player (a learner is ideal, but anyone who is not player 1 or 2), the same
piano and phone, and about 30 minutes. That covers 31 takes of 10 to 25 seconds each, plus retries.

---

## Before you start

- **Phone:** in the same spot as the earlier recordings, about 1 m from the keyboard, mic facing
  the strings. Do not move it between takes.
- **No sustain pedal.** Pedal is a separate problem; keep it out of this set.
- **Silence:** leave about 2 seconds of silence before the first note and after the last one.
  Nobody speaks during a take; the take name goes in the file name, not on the recording.
- **Tempo:** use a comfortable, steady tempo unless the take says otherwise. No metronome click
  audible in the recording.
- **One planned mistake per take**, played naturally. Then keep going, as a student would.
- **If the take comes out different from the plan, keep it.** Label what was actually played, not
  what the shot list says. If an unplanned mistake slips in, keep that take too, and label both.
  Only re-record if the take is unusable (phone knocked, someone talking).

## File names

`3<piece>-<hands>-<what>.m4a`, saved to `recordings/eval-interviews/raw/`. The names below already
follow the intake rules. The part after the hands (`clean1`, `wrong3`, ...) is free text; do not add
the word `down` to anything except the descending E major takes.

---

## Words used below

- **Wrong pitch:** the wrong key is played instead of the written note.
- **Missed:** the written note is never played.
- **Extra:** a note the score does not have, most often the same key struck again.
- **Held too long:** the key stays down longer than written. Two kinds are recorded on purpose,
  because the existing labels do not make clear which one they mean:
  - *overlap:* the next note comes **on time**, but the previous key is still held, so both sound.
  - *stretch:* the note is held past its length, so the next note comes **after** its beat.
- **Late:** the previous key was released and there is a **silence** before the next note.

---

## A. Ode to Joy, melody only (`ode_melody`), 16 takes

Score (one bar per row; note ids in brackets):

| Bar | | | | |
|---|---|---|---|---|
| 1 | E4 (RH1) | E4 (RH2) | F4 (RH3) | G4 (RH4) |
| 2 | G4 (RH5) | F4 (RH6) | E4 (RH7) | D4 (RH8) |
| 3 | C4 (RH9) | C4 (RH10) | D4 (RH11) | E4 (RH12) |
| 4 | E4. (RH13) | D4 eighth (RH14) | D4 half (RH15) | |

| # | File | What to play | Expected label |
|---|---|---|---|
| A1 | `3ode-1h-clean1` | Clean, normal tempo | perfect |
| A2 | `3ode-1h-clean2` | Clean, slow (about half speed) | perfect |
| A3 | `3ode-1h-clean3` | Clean, quicker than comfortable | perfect |
| A4 | `3ode-1h-wrong3` | Bar 1, beat 3: **F#4** (black key) instead of F4 | RH3 wrong_pitch, played F#4 |
| A5 | `3ode-1h-wrong11` | Bar 3, beat 3: **E4** instead of D4 | RH11 wrong_pitch, played E4 |
| A6 | `3ode-1h-fix6` | Bar 2, beat 2: hit **E4**, then quickly correct to F4, carry on | RH6 wrong_pitch, played E4 |
| A7 | `3ode-1h-shift` | Whole of bar 2 **one key too high**: A G F E instead of G F E D | RH5 to RH8 wrong_pitch, played A4 G4 F4 E4 |
| A8 | `3ode-1h-miss8` | Leave out bar 2 beat 4 (D4): go E, then straight to C | RH8 missed |
| A9 | `3ode-1h-miss10` | Bar 3: play **one** C instead of two, then carry on | RH10 missed |
| A10 | `3ode-1h-extra4` | Bar 1, beat 4: strike G4 **twice**, then the bar 2 G as written | extra after RH4, played G4 |
| A11 | `3ode-1h-overlap7` | Bar 2: keep the **E4 key held down** while playing the D4, in time; let go about half a beat later | RH7 held_long |
| A12 | `3ode-1h-stretch12` | Bar 3, beat 4: hold the E4 for **two beats**, then carry on from bar 4 | RH12 held_long |
| A13 | `3ode-1h-pause9` | End of bar 2: **lift your hand**, wait about one beat in silence, then start bar 3 | RH9 late |
| A14 | `3ode-1h-restart5` | Play up to bar 2 beat 3 (E4), stop, start again from **bar 2 beat 1**, play to the end | perfect; comment "restart RH7 -> RH5" |
| A15 | `3ode-1h-restart1` | Play up to bar 3 beat 3 (D4), stop, start again **from the beginning**, play to the end | perfect; comment "restart RH11 -> RH1" |
| A16 | `3ode-1h-stop` | Stop after bar 3 beat 3 (D4) and do not finish | RH12 to RH15 missed |

## B. C major scale, right hand (`c_major_rh`), 7 takes

C4 D4 E4 F4 | G4 A4 B4 C5 (RH1 to RH8)

| # | File | What to play | Expected label |
|---|---|---|---|
| B1 | `3cmajor-1h-clean1` | Clean, normal tempo | perfect |
| B2 | `3cmajor-1h-clean2` | Clean, fast | perfect |
| B3 | `3cmajor-1h-wrong5` | **A4** instead of G4 | RH5 wrong_pitch, played A4 |
| B4 | `3cmajor-1h-miss4` | Leave out F4 (E straight to G) | RH4 missed |
| B5 | `3cmajor-1h-extra3` | Strike E4 **twice** | extra after RH3, played E4 |
| B6 | `3cmajor-1h-overlap6` | Keep **A4 held** while playing B4, in time | RH6 held_long |
| B7 | `3cmajor-1h-restart` | Play up to A4, stop, start again **from C4**, play to the top | perfect; comment "restart RH6 -> RH1" |

## C. E major scale, right hand, descending (`e_major_rh_down`), 3 takes

E5 D#5 C#5 B4 | A4 G#4 F#4 E4 (RH1 to RH8). Forgetting a sharp is the most common scale mistake.

| # | File | What to play | Expected label |
|---|---|---|---|
| C1 | `3emajor-1h-down-clean` | Clean | perfect |
| C2 | `3emajor-1h-down-wrong3` | **C5** (white key) instead of C#5 | RH3 wrong_pitch, played C5 |
| C3 | `3emajor-1h-down-wrong6` | **G4** (white key) instead of G#4 | RH6 wrong_pitch, played G4 |

## D. Two hands, 5 takes (secondary, not part of the 0.9 target)

| # | File | What to play | Expected label |
|---|---|---|---|
| D1 | `3ode-2h-clean` | Ode with LH bass, clean | perfect |
| D2 | `3ode-2h-wronglh2` | Bar 2 left hand: **A3** instead of G3 | LH2 wrong_pitch, played A3 |
| D3 | `3ode-2h-missrh9` | Bar 3 beat 1: play the LH C3 but **leave out the RH C4** | RH9 missed (v3's known blind spot: this measures it) |
| D4 | `3cmajor-2h-clean` | C major hands together, clean | perfect |
| D5 | `3cmajor-2h-wrong5` | Right hand **A4** instead of G4 (LH plays G3 as written) | RH5 wrong_pitch, played A4 |

---

## After recording

1. Register the takes **as test** (this keeps them out of every tuning run):
   ```bash
   .venv/bin/python -m eval.cli sync --dry-run --split test
   .venv/bin/python -m eval.cli sync --split test
   ```
   The dry run should list all 31 takes with the piece shown in the tables above.
2. Label them in the shared sheet as usual (`todo` -> `correct`, then mark the mistakes).
   Listen to every take; the "expected label" column is only a guide. For restart takes, label the
   **final attempt** and put the restart in the comment, for example `restart RH7 -> RH5`.
3. Score them on their own, once:
   ```bash
   .venv/bin/python -m eval.cli evaluate --sheet ~/Downloads/labels.csv --model v3 --split test
   ```

**Rule for the test split:** look at the score, then do not change v3 because of it. Any fix
made after seeing these results needs another fresh set to be measured on.
