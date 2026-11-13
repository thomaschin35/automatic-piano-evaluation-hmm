# Recording shot list, round 2: held-out test set (one hand only)

**Purpose:** v3 is now being tuned on the earlier rounds, including round 1 (player 3, file prefix
`3`). Those takes no longer measure it fairly, so round 2 is the new held-out test set: nobody looks
at these takes while tuning. Round 1 showed where v3 is weak on one-hand pieces, and this round
records more of exactly those cases: **extra** notes (above all the same key struck again while it
is still ringing, v3's biggest blind spot), **missed** notes inside repeated pairs, **late** notes,
and **soft** playing. Soft notes matter because a quiet note is easy for the model to lose under the
ringing of the one before. Repeated notes matter because a second strike of a ringing key changes
the sound only a little, so the model must notice the new attack.

**What you need:** the same piano and phone as before, and about 45 minutes. That covers 46 takes of
10 to 25 seconds each, plus retries.

**If you are short on time,** record the groups in this order and stop wherever you have to:
Extra, Late, Missed, Wrong pitch, Clean, Restart, Combination.

## Balance

| Type | Takes | Ode melody | C major RH | E major down |
|---|---|---|---|---|
| clean (no mistake) | 6 | 4 | 1 | 1 |
| extra | 9 | 6 | 2 | 1 |
| missed | 8 | 6 | 1 | 1 |
| wrong pitch | 8 | 4 | 2 | 2 |
| late | 9 | 4 | 2 | 3 |
| restart (goes back) | 3 | 1 | 1 | 1 |
| combination (two mistakes) | 3 | 1 | 1 | 1 |
| **total** | **46** | **26** | **10** | **10** |

Notes are not quite balanced: take 22 has two missed notes, take 40 has two late notes, and each
combination take has two mistakes.

---

## Before you start

- **One hand only.** Only these three pieces: Ode to Joy melody, C major right hand going **up**,
  E major right hand going **down**. No two-hand takes, no Canon.
- **Phone:** in the same spot as the earlier recordings, about 1 m from the keyboard, mic facing
  the strings. Do not move it between takes.
- **No sustain pedal.** Pedal is a separate problem; keep it out of this set.
- **Silence:** leave about 2 seconds of silence before the first note and after the last one.
  Nobody speaks during a take; the take name goes in the file name, not on the recording.
- **Tempo:** use a comfortable, steady tempo unless the take says otherwise. No metronome click
  audible in the recording.
- **Loudness:** play at a normal, medium loudness unless the take says soft or loud. "Soft" means
  clearly quieter than normal; "very soft" means as quiet as the piano still sounds.
- **One planned mistake per take** (two in the Combination group), played naturally. Then keep
  going, as a student would.
- **Do not hold a key down under the next note** (two keys sounding together) unless a take asks
  for it. Overlapping keys are not part of this round.
- **If the take comes out different from the plan, keep it, and tell us what was actually played.**
  It gets labelled with what was played, not what the shot list says. If an unplanned mistake slips
  in, keep that take too and mention both. Only re-record if the take is unusable (phone knocked,
  someone talking).
- **Shuffle the order** if you can, so you are not doing nine of the same mistake in a row.

## File names

`4<piece>-1h-<what>.m4a`, saved to `recordings/round2/`. The piece words are the same as round 1:
`4ode-1h-...`, `4cmajor-1h-...`, `4emajor-1h-down-...`. Use the names in the tables below exactly;
each one is unique. Only the descending E major takes contain the word `down`.

---

## Words used below

- **correct:** the written note, played on time.
- **wrong_pitch:** the wrong key is played instead of the written note. The label gives the key
  that was played: "RH3 wrong_pitch, played G4".
- **missed:** the written note is never played.
- **extra:** a note the score does not have, either the same key struck again or a different key.
  The label says which written note it comes after and what key was played: "extra after RH4,
  played G4". A stray note before the first written note is "extra after START".
- **late:** the note **started** late, whatever the reason. Either the player lifted the previous
  key and waited (a silence), or kept the previous key down and waited (the previous note rings on).
  Both are labelled `late` on the note that arrives late. There is **no "held too long" label** in
  this round.
- **Restart:** the player stops and goes back to an earlier note, then plays on. Label the **final
  attempt** only, and put the restart in the comment: "restart RH7 -> RH5".

---

## Scores (note ids)

**Ode to Joy, melody** (`ode_melody`), one bar per row:

| Bar | beat 1 | beat 2 | beat 3 | beat 4 |
|---|---|---|---|---|
| 1 | E4 (RH1) | E4 (RH2) | F4 (RH3) | G4 (RH4) |
| 2 | G4 (RH5) | F4 (RH6) | E4 (RH7) | D4 (RH8) |
| 3 | C4 (RH9) | C4 (RH10) | D4 (RH11) | E4 (RH12) |
| 4 | E4 dotted (RH13) | | D4 eighth (RH14, beat 2.5) | D4 half (RH15, beat 3) |

**C major, right hand, going up** (`c_major_rh`): C4 D4 E4 F4 | G4 A4 B4 C5 (RH1 to RH8).

**E major, right hand, going down** (`e_major_rh_down`): E5 D#5 C#5 B4 | A4 G#4 F#4 E4 (RH1 to RH8).

---

## Clean, 6 takes

| # | File | What to play | Expected label |
|---|---|---|---|
| 1 | `4ode-1h-clean-normal` | Ode melody, normal tempo, normal loudness | perfect |
| 2 | `4ode-1h-clean-verysoft` | Ode melody, **very soft**: as quiet as the piano still sounds, every note | perfect |
| 3 | `4ode-1h-clean-loud` | Ode melody, **loud** (but not banged) | perfect |
| 4 | `4ode-1h-clean-fast` | Ode melody, **fast**: quicker than comfortable, still steady | perfect |
| 5 | `4cmajor-1h-clean-soft` | C major going up, **soft** | perfect |
| 6 | `4emajor-1h-down-clean-soft` | E major going down, **soft** | perfect |

## Extra, 9 takes

Takes 7 to 12 strike the **same key again**. "Full beat" means the extra strike takes a beat of its
own and everything after it comes one beat later. "Squeezed" means the two strikes fit inside one
beat (like two eighth notes) and the next written note still comes on time.

| # | File | What to play | Expected label |
|---|---|---|---|
| 7 | `4ode-1h-extra-e2-triple` | Ode bar 1: strike E4 **three times** where two are written (E E E F G). Full beat. | extra after RH2, played E4 |
| 8 | `4ode-1h-extra-c10-soft` | Ode bar 3: strike C4 **three times** where two are written; the third C is played **soft**. Full beat. | extra after RH10, played C4 |
| 9 | `4ode-1h-extra-d15-quick` | Ode bar 4, beat 3: strike the D4 half note **twice quickly** (two eighths on beat 3), hold the second to the end. Squeezed. | extra after RH15, played D4 |
| 10 | `4cmajor-1h-extra-g5-quick` | C major: strike G4 **twice quickly** inside its beat; A4 comes on time. Squeezed. | extra after RH5, played G4 |
| 11 | `4emajor-1h-down-extra-b4-soft` | E major: strike B4 **twice**, the second one **soft**. Full beat. | extra after RH4, played B4 |
| 12 | `4ode-1h-extra-f6` | Ode bar 2, beat 2: strike F4 **twice** (G F F E D). Full beat. | extra after RH6, played F4 |
| 13 | `4ode-1h-extra-brush-a` | Ode bar 1 to bar 2: between the G4 on beat 4 and the G4 on bar 2 beat 1, lightly **brush A4** (the white key above) by accident. Squeezed: the bar 2 G4 stays on time. | extra after RH4, played A4 |
| 14 | `4cmajor-1h-extra-fsharp` | C major: between F4 and G4, quickly touch **F#4** (the black key between them), then G4 on time. Squeezed. | extra after RH4, played F#4 |
| 15 | `4ode-1h-extra-start` | Ode: strike a stray **F4** by accident, wait about a beat, then play the whole melody as written from E4 | extra after START, played F4 |

## Missed, 8 takes

Takes 16 to 19 leave out one note of a **repeated pair**: play the key once where it is written
twice. "Keep the beat" means you hold the single note for both beats, so the next note comes on
time. "Drop the beat" means you go straight on to the next note, so everything after it comes one
beat earlier. In all four the label is `missed` on the **second** note of the pair.

| # | File | What to play | Expected label |
|---|---|---|---|
| 16 | `4ode-1h-miss-e2-keep` | Ode bar 1: play **one** E4, hold it for two beats, then F4 on beat 3. Keep the beat. | RH2 missed |
| 17 | `4ode-1h-miss-g5-drop` | Ode bar 1 to bar 2: play **one** G4 (beat 4), then go straight to F4 (E E F G F E D). Drop the beat. | RH5 missed |
| 18 | `4ode-1h-miss-c10-keep` | Ode bar 3: play **one** C4, hold it for two beats, then D4 on beat 3. Keep the beat. | RH10 missed |
| 19 | `4ode-1h-miss-e13-drop` | Ode bar 3 to bar 4: play **one** E4 (one beat), then go straight to the D4 eighth and the D4 half (D E D D). Drop the beat. | RH13 missed |
| 20 | `4ode-1h-miss-f6` | Ode bar 2: leave out the F4 on beat 2; go G, then straight to E D (G E D). | RH6 missed |
| 21 | `4emajor-1h-down-miss-gsharp6` | E major: leave out G#4; go A4 straight to F#4 | RH6 missed |
| 22 | `4cmajor-1h-miss-e3f4` | C major: leave out **E4 and F4**; go D4 straight to G4 | RH3 missed, RH4 missed |
| 23 | `4ode-1h-miss-first` | Ode: leave out the very first E4; start on the **second** E4 (one E, then F G G ...) | RH1 missed |

## Wrong pitch, 8 takes

| # | File | What to play | Expected label |
|---|---|---|---|
| 24 | `4ode-1h-wrong-dsharp11-soft` | Ode bar 3, beat 3: play **D#4** (black key) instead of D4, and play it **soft** | RH11 wrong_pitch, played D#4 |
| 25 | `4emajor-1h-down-wrong-d2` | E major: **D5** (white key) instead of D#5, the forgotten sharp | RH2 wrong_pitch, played D5 |
| 26 | `4emajor-1h-down-wrong-g6` | E major: **G4** (white key) instead of G#4, the forgotten sharp | RH6 wrong_pitch, played G4 |
| 27 | `4ode-1h-wrong-anticip3` | Ode bar 1, beat 3: play **G4** instead of F4, jumping ahead (E E G G, then bar 2 as written) | RH3 wrong_pitch, played G4 |
| 28 | `4ode-1h-wrong-repeat3` | Ode bar 1, beat 3: play **E4** again instead of F4 (E E E G, then bar 2 as written) | RH3 wrong_pitch, played E4 |
| 29 | `4ode-1h-wrong-fix12` | Ode bar 3, beat 4: hit **F4**, quickly correct to E4, carry on | RH12 wrong_pitch, played F4 |
| 30 | `4cmajor-1h-wrong-b6` | C major: play **B4** instead of A4 (C D E F G B B C) | RH6 wrong_pitch, played B4 |
| 31 | `4cmajor-1h-wrong-octave1` | C major: start on **C5** (an octave too high) instead of C4, then move the hand down and carry on from D4 | RH1 wrong_pitch, played C5 |

## Late, 9 takes

"Lift" means take the finger off and wait in silence. "Keep down" means keep the previous key held
for the extra time, then play the next note (only one key down at a time). Both are labelled `late`
on the next note.

| # | File | What to play | Expected label |
|---|---|---|---|
| 32 | `4ode-1h-late-lift14` | Ode bar 4: play the dotted E4 its full length, **lift**, wait about one beat in silence, then the D4 eighth and the D4 half | RH14 late |
| 33 | `4cmajor-1h-late-lift7` | C major: after A4, **lift**, wait about one beat, then B4 | RH7 late |
| 34 | `4emajor-1h-down-late-lift3` | E major: after D#5, **lift**, wait about one beat, then C#5 | RH3 late |
| 35 | `4ode-1h-late-keep7` | Ode bar 2: **keep F4 down** for two beats instead of one, then E4 | RH7 late |
| 36 | `4cmajor-1h-late-keep4` | C major: **keep E4 down** for two beats instead of one, then F4 | RH4 late |
| 37 | `4emajor-1h-down-late-keep6` | E major: **keep A4 down** for two beats instead of one, then G#4 | RH6 late |
| 38 | `4ode-1h-late-half11` | Ode bar 3: after the second C4, **lift**, wait only about **half a beat**, then D4 | RH11 late |
| 39 | `4emajor-1h-down-late-half7` | E major: after G#4, **lift**, wait only about **half a beat**, then F#4 | RH7 late |
| 40 | `4ode-1h-late-two3-10` | Ode: two pauses. Bar 1: after the second E4, **lift**, wait about one beat, then F4. Bar 3: after the first C4, **lift**, wait about one beat, then the second C4. | RH3 late, RH10 late |

## Restart, 3 takes

Label the **final attempt** only, and put the restart in the comment (there is no restart label
type).

| # | File | What to play | Expected label |
|---|---|---|---|
| 41 | `4ode-1h-restart-wrong7` | Ode bar 2: play G4 F4, then hit **D#4** instead of E4 on beat 3, stop, start again from **bar 2 beat 1** (G4), play to the end | perfect; comment "restart RH7 -> RH5, first try at RH7 played D#4" |
| 42 | `4cmajor-1h-restart-back2` | C major: play up to B4, stop, go back **two notes** to G4, play G A B C to the top | perfect; comment "restart RH7 -> RH5" |
| 43 | `4emajor-1h-down-restart-pause` | E major: play E5 D#5 C#5 B4, stop, **pause about one beat**, start again **from the beginning** (E5), play to the end | perfect; comment "restart RH4 -> RH1" |

## Combination, 3 takes

Two different mistakes, far apart in the same take. Everything else as written.

| # | File | What to play | Expected label |
|---|---|---|---|
| 44 | `4ode-1h-combo-miss3-late9` | Ode bar 1: leave out the F4 on beat 3 (lift and leave that beat silent; G4 on beat 4). Then at the end of bar 2: **keep D4 down** for two beats instead of one, then C4 | RH3 missed, RH9 late |
| 45 | `4cmajor-1h-combo-extra2-wrong7` | C major: strike D4 **twice** (full beat). Later, play **A#4** (black key) instead of B4 | extra after RH2, played D4; RH7 wrong_pitch, played A#4 |
| 46 | `4emajor-1h-down-combo-wrong3-miss7` | E major: play **D5** (white key) instead of C#5. Later, leave out F#4 (G#4 straight to E4) | RH3 wrong_pitch, played D5; RH7 missed |

---

## After recording

Put the files in `recordings/round2/` and tell Claude they are ready. Claude registers the takes,
labels them from this list and scores them.
