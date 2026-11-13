# Recording shot list, round 3: pieces the model has never seen

**Purpose:** every setting in v3 was chosen on three short tunes (Ode to Joy, C major, E major).
Round 3 checks that it works on other music without anyone adjusting it. All takes are **played
correctly**, so there is nothing to label: anything v3 flags on these takes is a false alarm.

**What you need:** the same phone, about 10 minutes, 16 takes of 10 to 30 seconds each.
One hand, steady tempo, start recording about a second before the first note.

**Where the files go:** `recordings/round3/`, named exactly as in the table (`.m4a`). The prefix `5`
marks round 3. Then run `python -m eval.cli evaluate`: it registers them, and because every take is
clean you can mark each one perfect in the label sheet without listening again.

## The pieces

| Piece | Notes to play | What it tests |
|---|---|---|
| **Twinkle** | C C G G A A G(long) / F F E E D D C(long) | repeated notes, a jump of five keys |
| **Hot Cross Buns** | E D C(long) / E D C(long) / C C C C D D D D (twice as fast) / E D C(long) | fast repeated notes |
| **Frere Jacques** | C D E C, C D E C / E F G(long), E F G(long) / G A G F (fast) E C, G A G F (fast) E C / C G(below) C(long), C G(below) C(long) | a longer piece (32 notes), fast notes, a low note |
| **Low C scale** | C3 up to C4 and back down to C3 (one octave below middle C) | low keys |
| **High C scale** | C5 up to C6 and back down to C5 (one octave above middle C) | high keys |
| **Rests** | C, rest, E, rest / G, rest, high C(long) / G, rest, E, rest / C(hold 4 beats). Lift the finger for each rest. | written silences |

All start on middle C (C4) unless the row says otherwise. "Long" is two beats.

## The takes

| # | File name | How to play |
|---|---|---|
| 1 | `5twinkle-1h-clean1` | normal |
| 2 | `5twinkle-1h-clean2` | normal |
| 3 | `5twinkle-1h-slow` | clearly slower than normal |
| 4 | `5twinkle-1h-fast` | clearly faster than normal |
| 5 | `5buns-1h-clean1` | normal |
| 6 | `5buns-1h-clean2` | normal |
| 7 | `5buns-1h-soft` | as quietly as you can while every note still sounds |
| 8 | `5frere-1h-clean1` | normal |
| 9 | `5frere-1h-clean2` | normal |
| 10 | `5frere-1h-soft` | quietly |
| 11 | `5clow-1h-clean1` | normal |
| 12 | `5clow-1h-clean2` | normal |
| 13 | `5chigh-1h-clean1` | normal |
| 14 | `5chigh-1h-soft` | quietly |
| 15 | `5rests-1h-clean1` | normal |
| 16 | `5rests-1h-clean2` | normal |

**If you are short on time,** takes 1, 5, 8, 11, 13 and 15 (one per piece) are enough for a first answer.

If you can, record a few of these on a **different piano or phone** as well (same names with a
`6` prefix). That is the one thing none of the existing recordings can tell us.
