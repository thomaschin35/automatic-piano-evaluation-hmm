# Hand-off: v2 alignment collapses on fast playing

From the evaluation side. These are findings only; the evaluation code has not changed and no model code was touched.

## Symptom
Two takes are played correctly but get almost every note flagged (eval run `20261029-111844_v2_0e0f184-dirty`):

| take | piece | gold | v2 prediction |
|---|---|---|---|
| `1cmajor-up` | `c_major_rh` | perfect | 7 of 8 notes wrong (RH1 missed; RH2–RH7 wrong_pitch or missed) |
| `1emajor-1h-down` | `e_major_rh_down` | perfect | 6 of 8 notes wrong |

Decoded onsets for `1cmajor-up` (original-file seconds):

```
RH1 1.28 | RH2 2.30 | RH3 2.44 | RH4 2.58 | RH5 2.72 | RH6 2.86 | RH7 3.02 | RH8 3.34
true:  C4 1.30 | D4 1.56 | E4 1.76 | F4 2.03 | G4 2.35 | A4 2.61 | B4 2.83 | C5 3.04   (pyin)
```

The Viterbi path keeps RH1 for about 1.0 s, about 4 real notes. It then runs RH2–RH7 through at the minimum chain length of about 3 frames each, so every grading window sits on the neighbouring note. The "played" pitch reported is usually the next scale note.

## Ruled out
- **Leading silence or trimming.** `librosa.effects.trim(top_db=25)` starts within 0.02 s of the first note: 1.28 vs 1.30 s, and 0.88 vs 0.89 s. Cropping the silence off the wav by hand gives an identical result, just shifted in time.
- **Tempo prior.** `create_initial_alignment` scales the expected durations to the trimmed length, so the expected durations are about right: about 0.27 s, or about 12 frames per note.
- **Audio quality.** pyin tracks all 8 notes cleanly and in order. `onset_any` has a clear burst at every note.

## Likely cause
The player plays about 0.23–0.27 s per note. At `seconds_per_frame = 0.023` that is about 10–12 frames per note. The `onset_any` bursts are about **5 frames wide**: the 4096-sample (186 ms) window smears them, and the ±1-frame max-pool widens them further. Example from `1cmajor-up`:

```
t    1.28 1.30 1.32 1.35 1.37 1.39 1.42 1.44 1.46 1.49 1.51 1.53 1.56 1.58
ons  0.97 0.99 0.99 0.99 0.97 0.37 0.06 0.06 0.95 0.98 0.98 0.98 0.96 0.48
```

In `build_graph` / `log_emissions`, the attack chain is 1 frame (`attack_frames=1`). Every `min`/`free` frame, and the lead-in, pays `onset_weight * log(eps + (1-eps)(1-o))`, which is about −10 per frame when o ≈ 0.99. So for a fast note, about 4 of its roughly 10 frames carry a heavy penalty in any state except one attack frame. Lining each note up with its own burst stops being clearly better than smearing one note across several bursts. The bounded pitch term (`pitch_floor=0.6`, at most about −0.5 per frame) is too weak to pull the path back.

The slower takes of the same pieces align perfectly. `2cmajor-1h-up` plays about 0.5 s per note and gets 8 of 8 correct. Player 1's false alarms on Ode RH2/RH3 look like the same failure.

## Ideas (yours to judge)
- Make the attack chain about as wide as an onset burst, for example scaled to the STFT window. Alternatively, soften the "no onset" penalty for the first few frames after an attack.
- Peak-pick `onset_any`, or drop the ±1 pad, so a burst is 1–2 frames wide.
- Use a shorter window or hop for the onset stream only.
- Give the pitch term more weight inside `min`/`free` states.

## How to check
```bash
.venv/bin/python -m eval.cli run --model v2 --piece c_major_rh e_major_rh_down
.venv/bin/python -m eval.cli compare <previous-run> <new-run>
```
Success means `1cmajor-up` and `1emajor-1h-down` drop to 0 false alarms, while `2cmajor-1h-up` and `1emajor-1h-up` stay clean. Then run the full dev split, because the Ode and Canon player-1 takes should improve as well.
