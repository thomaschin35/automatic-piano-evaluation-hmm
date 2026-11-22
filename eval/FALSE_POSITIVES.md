# False positives in v3: what causes them

Written 2026-11-22. Model `stft_hmm_v3.py` at `5440e5b` (current) and `c0e7216` (as scored blind on
round 3). A false positive (FP) = the model flags wrong / missed / extra / late where the label says
the note was fine, or names a real mistake with the wrong type.

## 1. The one-line answer

Almost every FP is **one key's sound credited to another key**. A piano note is a stack of partials
(octave, fifth, etc.), and v3 decides "key X was struck" from energy at X's frequencies. FPs happen
when that energy actually came from a different key: one that shares a partial, one still ringing,
one a semitone away in the bass, or the other hand. The rest are where two labels sound the same
(extra vs "wrong, then corrected") plus one stray noise.

## 2. The blind round-3 false alarms (36), by cause

Round 3 was the only blind test. Re-scored with the blind model (`c0e7216`); every one of the 36 traced.

| # | Cause | FPs | Examples | Now |
|---|---|---|---|---|
| 1 | **Phantom key**: a key a 4th/5th/octave from the struck key picks up their shared partial and looks struck | 14 | `5chigh` E5 read as B5 (4 takes); in `5frere` every G3 also "struck" C3 → C3 extra (8); twinkle C4 read as F4; B5 extra | fixed (phantom rule) |
| 2 | **Ringing bass erases the melody**: LH key held under an RH strike on its octave/harmonic twitches, and the RH note is filed as its overtone | 8 | `5twinkle-2h` RH9 F4 "played F3" (6 takes); C4 / G4 over a held C3 | fixed (ring rule) |
| 3 | **Extra before the beat read as "wrong, then corrected"** | 4 | B3 before Frere's first note; A4 in Ode; D4 in a rest | 3 fixed; `5chigh-1h-extra-g5` left |
| 4 | **Hands not together** → missed + extra instead of late | 4 | `5ode-2h-late-rh5`, `5twinkle-2h-late-lh2` | fixed |
| 5 | **Fast repeated note** (< 0.35 s apart): the re-strike shows only at the octave, or not at all | 2 | buns C4 → C5; `5ode-2h-fast` C4 missed | **open** |
| 6 | **Low register, descending semitone**: E3 right after F3 | 1 | `5clow-1h-clean` E3 → E4 | **open** |
| 7 | **Other hand strikes the sub-harmonic at the same moment** | 1 | `5twinkle-2h-extralh`: real C3 re-strike erases G4 (G4 = C3's 3rd partial) | **open** |
| 8 | **Stray key a 4th up, just under the phantom line** (9 dB gap, line is 12) | 1 | `5chigh-1h-miss6-extra13`: A#5 next to F5 reported as extra | **open** |
| 9 | **Lone low thump after the last note** | 1 | `5ode-2h-late-held9`: G2 extra 0.6 s after the end | **open** |

28 of the 36 were false alarms on correctly played notes; 8 (causes 3-4) were real mistakes given the wrong name.
The 7 FPs left today are exactly causes 3 (1), 5-9.

## 3. The open causes, measured

**5. Fast repeated notes.** Of 391 correct repeated notes (one hand + Ode two-hand), FPs happen only
below 0.35 s between strikes: 3 of 83 there (4%), 0 of 308 above. In `5buns-1h-wrong11` RH8 the C4's
own fundamental rose 1 dB (still ringing) while its octave C5 showed a full new attack. The
partial re-strike rule (`partial_restrike`) only accepts a partial *without* a full attack, so the
C5 stands as its own key → "wrong pitch, played C5". In `5ode-2h-fast` no event was detected at all.

**6. E3 after F3.** All six E3s in round 3, fundamental rise at the strike:

| | going up (after D3) | going down (after F3) |
|---|---|---|
| E3 rise | 32-37 dB | **10-11 dB** |
| E3 onset score | 0.60-0.80 | 0.34-0.61 |

F3 is a semitone up and still ringing; in the bass a semitone is under 3 FFT bins, so F3 leaks into
E3's bin and hides the attack. E4 (E3's octave) gets a clean attack and becomes the event's key.
The clean take's E3 fell under `onset_floor` (0.40); the other two passed narrowly (0.52, 0.61).
Expect the same for any descending semitone below about C4 (B2 after C3, etc.).

**7. Sub-harmonic struck by the other hand.** G4 0.89 and C3 0.86 struck together; G4 ≤ 1.1 × C3,
so G4 is folded in as C3's partial. C3 really rose 24 dB (not ringing), so the ring rule does not save
G4. The audio is ambiguous: G4 alone vs C3 alone would look similar, but here both keys were
written in the take (C3 just before) and G4 is due.

**3. `5chigh-1h-extra-g5`.** The tapped G5 landed on F5's beat and F5 came 1.3 beats late. That is
the same sound as "G5 wrong, then corrected to F5". A labelling boundary, not a detector failure.

**8 / 9.** One-offs: A#5 9 dB below F5 (phantom line 12 dB); G2 thump after the last note
(likely key release or handling noise).

## 4. Latent FP sources (correct today, close to the line)

Scanned every unwritten key beside the 2,000+ correct notes in all 153 takes.

- **C3's 7th partial (A#5) is not in `OVERTONE_OFFSETS`.** A#5 shows up beside correct notes 71
  times, 67 in round 3; 61 of those over a written C3 (7 × 130.8 Hz ≈ 916 Hz ≈ A#5). Offset 34 is
  missing from `(12, 19, 24, 28, 31, 36)`. Median strength 0.24, max 0.55, so mostly under the 0.40
  report cut. A louder bass, another piano or another phone could push it over → false extras /
  smudges on any two-hand piece with a C3 (or any bass key: its 7th partial is 2 octaves + a flat 7th up).
- **Phantom line (12 dB).** 16 related keys (4th/5th/octave) sit 6-12 dB below and are *not*
  tagged phantom. Today they are all written bass notes (C3 under C4) or harmless; cause 8 is one
  that wasn't.
- **Second-key extra rule (≥ 0.40, full attack).** 146 full-attack stray keys beside correct notes:
  4 at ≥ 0.40 (3 real extras + the cause-8 FP), 4 more at 0.30-0.40.

## 5. Where FPs concentrate

FPs come with: two hands (a bass key ringing under the melody), repeats faster than ~0.35 s, keys
outside C4-E5, and keys a 4th/5th/octave apart sounding together. The development takes have
almost none of these (one hand, middle register, moderate tempo, stepwise), which is why they score
0.99 and round 3 scored 0.61 blind.

## 6. How this was measured

Throwaway scripts (not in the repo): a full `eval.cli run` at `5440e5b` and again with the
`c0e7216` model swapped in, a per-event dump (strikes, salience, full attack, rise in dB, phantom
and overtone links, verdicts) for each FP, and a scan of every unwritten key beside gold-correct notes.
