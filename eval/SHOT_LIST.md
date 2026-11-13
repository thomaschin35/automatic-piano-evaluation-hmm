# Recording shot list: held-out test set (player 3)

**Purpose:** check v3 on a pianist it has never seen. Every number so far comes from players 1 and 2,
and v3's settings were chosen while looking at their takes. The scoreboard reports player 3 as its
own row, so the first run on these takes shows how v3 does on a new player.

**What you need:** a new player (a learner is ideal, but anyone who is not player 1 or 2), the same
piano and phone, and about 35 minutes. That covers 36 takes of 10 to 25 seconds each, plus retries.

## Balance

Each error type gets **5 takes**, spread across pieces so no type rests on one tune:

| Type | Takes | Ode melody | C major RH | E major down | Two hands |
|---|---|---|---|---|---|
| clean (no mistake) | 6 | 2 | 1 | 1 | 2 |
| wrong pitch | 5 | 3 | | 1 | 1 |
| missed | 5 | 3 | 1 | | 1 |
| extra | 5 | 2 | 1 | 1 | 1 |
| held too long | 5 | 2 | 2 | 1 | |
| late | 5 | 2 | 1 | 1 | 1 |
| restart (goes back) | 5 | 3 | 1 | 1 | |
| **total** | **36** | **17** | **7** | **6** | **6** |

Takes are balanced, notes are not quite: the hand-shift take has 4 wrong notes and the stop-early
take has 4 missed notes, because that is what those mistakes do.

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
- **Shuffle the order** if you can, so the player is not doing five of the same mistake in a row.

## File names

`3<piece>-<hands>-<what>.m4a`, saved to `recordings/eval-interviews/raw/`. The names below already
follow the intake rules. The part after the hands (`clean1`, `wrong3`, ...) is free text; do not add
the word `down` to anything except the descending E major takes.

---

## Words used below

- **Wrong pitch:** the wrong key is played instead of the written note.
- **Missed:** the written note is never played.
- **Extra:** a note the score does not have, most often the same key struck again.
- **Held too long:** the key stays down longer than written. Two kinds are recorded on purpose
  (3 and 2 takes), because the existing labels do not make clear which one they mean:
  - *overlap:* the next note comes **on time**, but the previous key is still held, so both sound.
  - *stretch:* the note is held past its length, so the next note comes **after** its beat.
- **Late:** the previous key was released and there is a **silence** before the next note.
- **Restart:** the player stops and goes back to an earlier note, then plays on.

---

## Scores (note ids)

**Ode to Joy, melody** (`ode_melody`), one bar per row:

| Bar | beat 1 | beat 2 | beat 3 | beat 4 |
|---|---|---|---|---|
| 1 | E4 (RH1) | E4 (RH2) | F4 (RH3) | G4 (RH4) |
| 2 | G4 (RH5) | F4 (RH6) | E4 (RH7) | D4 (RH8) |
| 3 | C4 (RH9) | C4 (RH10) | D4 (RH11) | E4 (RH12) |
| 4 | E4 dotted (RH13) | | D4 eighth (RH14, beat 2.5) | D4 half (RH15, beat 3) |

**Ode to Joy with bass** (`ode_harmony`): the same melody, with LH C3 (LH1, bar 1), G3 (LH2, bar 2),
C3 (LH3, bar 3), G3 (LH4, bar 4), each a half note on beat 1 followed by a half rest.

**C major, right hand** (`c_major_rh`): C4 D4 E4 F4 | G4 A4 B4 C5 (RH1 to RH8).

**C major, hands together** (`c_major_both`): RH as above; LH an octave lower, C3 to C4 (LH1 to LH8).

**E major, descending** (`e_major_rh_down`): E5 D#5 C#5 B4 | A4 G#4 F#4 E4 (RH1 to RH8).

---

## Clean, 6 takes

| # | File | What to play | Expected label |
|---|---|---|---|
| 1 | `3ode-1h-clean1` | Ode melody, normal tempo | perfect |
| 2 | `3ode-1h-clean2` | Ode melody, slow (about half speed) | perfect |
| 3 | `3cmajor-1h-clean` | C major, quicker than comfortable | perfect |
| 4 | `3emajor-1h-down-clean` | E major descending | perfect |
| 5 | `3ode-2h-clean` | Ode with bass | perfect |
| 6 | `3cmajor-2h-clean` | C major hands together | perfect |

## Wrong pitch, 5 takes

| # | File | What to play | Expected label |
|---|---|---|---|
| 7 | `3ode-1h-wrong3` | Ode bar 1, beat 3: **F#4** (black key) instead of F4 | RH3 wrong_pitch, played F#4 |
| 8 | `3ode-1h-fix6` | Ode bar 2, beat 2: hit **E4**, quickly correct to F4, carry on | RH6 wrong_pitch, played E4 |
| 9 | `3ode-1h-shift` | Ode, whole of bar 2 **one key too high**: A G F E instead of G F E D | RH5 to RH8 wrong_pitch, played A4 G4 F4 E4 |
| 10 | `3emajor-1h-down-wrong3` | E major: **C5** (white key) instead of C#5, the forgotten sharp | RH3 wrong_pitch, played C5 |
| 11 | `3cmajor-2h-wrong5` | C major hands together: right hand **A4** instead of G4 (LH plays G3) | RH5 wrong_pitch, played A4 |

## Missed, 5 takes

| # | File | What to play | Expected label |
|---|---|---|---|
| 12 | `3ode-1h-miss8` | Ode bar 2: leave out the D4 on beat 4; go E, then straight to C | RH8 missed |
| 13 | `3ode-1h-miss10` | Ode bar 3: play **one** C instead of two, then carry on | RH10 missed |
| 14 | `3ode-1h-stop` | Ode: stop after bar 3 beat 3 (D4) and do not finish | RH12 to RH15 missed |
| 15 | `3cmajor-1h-miss4` | C major: leave out F4 (E straight to G) | RH4 missed |
| 16 | `3ode-2h-missrh9` | Ode with bass, bar 3 beat 1: play the LH C3 but **leave out the RH C4** | RH9 missed (v3's known blind spot: this measures it) |

## Extra, 5 takes

| # | File | What to play | Expected label |
|---|---|---|---|
| 17 | `3ode-1h-extra4` | Ode bar 1, beat 4: strike G4 **twice**, then the bar 2 G as written | extra after RH4, played G4 |
| 18 | `3ode-1h-extra12` | Ode bar 3-4: strike E4 **three** times where two are written | extra after RH12, played E4 |
| 19 | `3cmajor-1h-extra3` | C major: strike E4 **twice** | extra after RH3, played E4 |
| 20 | `3emajor-1h-down-extra2` | E major: strike D#5 **twice** | extra after RH2, played D#5 |
| 21 | `3ode-2h-extralh` | Ode with bass, bar 1 beat 3 (written rest): strike the LH **C3 again** | extra after LH1, played C3 |

## Held too long, 5 takes

| # | File | What to play | Expected label |
|---|---|---|---|
| 22 | `3ode-1h-overlap7` | Ode bar 2: keep the **E4 held** while playing the D4 on time; let go about half a beat later | RH7 held_long |
| 23 | `3cmajor-1h-overlap6` | C major: keep **A4 held** while playing B4 on time | RH6 held_long |
| 24 | `3emajor-1h-down-overlap4` | E major: keep **B4 held** while playing A4 on time | RH4 held_long |
| 25 | `3ode-1h-stretch12` | Ode bar 3, beat 4: hold the E4 for **two beats**, then carry on from bar 4 | RH12 held_long |
| 26 | `3cmajor-1h-stretch4` | C major: hold F4 for **two beats**, then carry on | RH4 held_long |

## Late, 5 takes

| # | File | What to play | Expected label |
|---|---|---|---|
| 27 | `3ode-1h-pause9` | Ode, end of bar 2: **lift your hand**, wait about one beat in silence, then start bar 3 | RH9 late |
| 28 | `3ode-1h-pause4` | Ode bar 1: after F4, **lift**, wait about one beat, then G4 | RH4 late |
| 29 | `3cmajor-1h-pause5` | C major: after F4, **lift**, wait about one beat, then G4 | RH5 late |
| 30 | `3emajor-1h-down-pause5` | E major: after B4, **lift**, wait about one beat, then A4 | RH5 late |
| 31 | `3ode-2h-pause9` | Ode with bass: after bar 2, **lift both hands**, wait about one beat, then bar 3 | RH9 late, LH3 late |

## Restart, 5 takes

Label the **final attempt** only, and put the restart in the comment (there is no restart label
type yet).

| # | File | What to play | Expected label |
|---|---|---|---|
| 32 | `3ode-1h-restart5` | Ode: play to bar 2 beat 3 (E4), stop, start again from **bar 2 beat 1**, play to the end | perfect; comment "restart RH7 -> RH5" |
| 33 | `3ode-1h-restart1` | Ode: play to bar 3 beat 3 (D4), stop, start again **from the beginning**, play to the end | perfect; comment "restart RH11 -> RH1" |
| 34 | `3ode-1h-restart6` | Ode: play to bar 2 beat 3 (E4), go back **one note** to F4, play to the end | perfect; comment "restart RH7 -> RH6" |
| 35 | `3cmajor-1h-restart` | C major: play to A4, stop, start again **from C4**, play to the top | perfect; comment "restart RH6 -> RH1" |
| 36 | `3emajor-1h-down-restart` | E major: play to F#4, stop, start again from **A4** (bar 2), play to the end | perfect; comment "restart RH7 -> RH5" |

---

## After recording

1. Register the takes (the recordings live in `recordings/person3/`):
   ```bash
   .venv/bin/python -m eval.cli sync
   ```
   It should list all 36 takes with the piece shown in the tables above.
2. Label them in the shared sheet as usual (`todo` -> `correct`, then mark the mistakes).
   Listen to every take; the "expected label" column is only a guide.
3. Score everything:
   ```bash
   .venv/bin/python -m eval.cli evaluate --sheet ~/Downloads/labels.csv --model v3
   ```
   The `mono_p3` row of `python -m eval.cli check` is player 3 on their own. Note those numbers before
   changing v3 because of them: once v3 is tuned on player 3, that row no longer measures a new player.
