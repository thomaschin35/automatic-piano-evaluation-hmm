# v3 work log (branch `thomas-hmm-2`)

Goal: label every note correct / wrong pitch / missed / held too long with F1 >= 0.9 on monophonic
pieces, then follow backtracking. "goal F1" below = strict note labels + held_long, monophonic real takes.

Each entry is the scoreboard from `python -m eval.cli check` at the moment a change was accepted.
Decisions and failed attempts are noted in prose above the scoreboard they led to.

## 2026-11-10 17:29  Phase 0: v2 baseline on the new scoreboard

```
SCOREBOARD  model=v2  (59.0s)

REAL TAKES        takes notes | goal F1 | strict P     R    F1 | wrong missed extra  held  late | clean
  all                32   814 |   0.66  |       0.67  0.70  0.69 |  0.83  0.36   -    0.00  0.00 | 12/17
  mono               22   477 |   0.80  |       0.92  0.79  0.85 |  0.89  0.40   -    0.00  0.00 | 11/12
  two_hand           10   337 |   0.53  |       0.48  0.60  0.54 |  0.71  0.36   -    0.00   -   | 1/5
  mono_p1            10   227 |   0.95  |       0.95  0.95  0.95 |  1.00  0.00   -     -    0.00 | 5/6
  mono_p2            10   234 |   0.31  |       0.67  0.29  0.40 |  0.29  0.67   -    0.00  0.00 | 5/5
  goal counts (mono): strict tp/fp/fn=22/2/6  held_long tp/fp/fn=0/0/3  late=0/0/5  extra=0/0/0

ONSET-ONLY (pitch term off): 11/12 clean monophonic takes still flag nothing

SYNTHETIC (clean mono takes)  cases | exact | strict P     R    F1 | extra R | false flags/case
  clean                          12 |  0.92 |       0.00   -    0.00 |    -    | 0.08
  sub                           552 |  0.88 |       0.90  0.93  0.92 |    -    | 0.10
  shift                          66 |  0.14 |       0.61  0.15  0.24 |    -    | 0.33
  miss_ins_step                 126 |  0.48 |       0.36  0.48  0.41 |    -    | 0.87
  miss_ins_repeat               126 |  0.07 |       0.12  0.07  0.09 |    -    | 0.52
  miss_del                      114 |  0.28 |       0.31  0.28  0.29 |    -    | 0.63
  extra_score                   114 |  0.00 |       0.00   -    0.00 |   0.00  | 0.18
  extra_dup                     114 |  0.00 |       0.00   -    0.00 |   0.00  | 0.05
  hesitate                      126 |  0.92 |       0.00   -    0.00 |    -    | 0.08
  stop_early                     24 |  0.08 |       0.30  0.45  0.36 |    -    | 4.08
  restart                       258 |  0.65 |       0.00   -    0.00 |    -    | 0.35 | restart located  0.00

REAL-TAKE DISAGREEMENTS (37)
  1can-1h-2 LH37 A2 gold=correct+late pred=correct
  1can-2h-2 RH2 E5 gold=correct pred=missed
  1can-2h-2 LH55 D3 gold=wrong_pitch pred=correct
  1can-2h-2 RH8 C#5 gold=correct pred=missed
  1cmajor-2h-up RH2 D4 gold=correct pred=missed
  1cmajor-2h-up RH5 G4 gold=correct pred=missed
  2can-1h-1 LH41 B2 gold=correct+held_long pred=correct
  2can-1h-1 LH53 D2 gold=correct+late pred=correct
  2can-2h-1 RH1 F#5 gold=correct pred=missed
  2can-2h-1 RH2 E5 gold=correct pred=missed
  2can-2h-1 RH4 C#5 gold=correct pred=wrong_pitch
  2can-2h-1 LH53 D2 gold=wrong_pitch pred=missed
  2can-1h-2 LH13 F#2 gold=wrong_pitch pred=correct
  2can-1h-2 LH21 D2 gold=correct+late pred=correct
  2can-1h-2 LH29 A2 gold=correct+held_long pred=correct
  2can-1h-2 LH42 F#3 gold=correct+held_long pred=correct
  2can-1h-2 LH61 A2 gold=correct+late pred=correct
  2can-2h-2 LH21 D2 gold=wrong_pitch+held_long pred=missed
  2can-2h-2 LH22 A2 gold=wrong_pitch pred=correct
  2can-2h-2 LH23 D3 gold=wrong_pitch pred=missed
  2can-2h-2 LH24 F#3 gold=wrong_pitch pred=missed
  2can-2h-2 LH29 A2 gold=missed pred=correct
  2can-2h-2 LH30 E3 gold=missed pred=correct
  2can-2h-2 LH31 A3 gold=missed pred=correct
  2can-2h-2 LH32 C#4 gold=missed pred=correct
  2can-2h-2 RH2 E5 gold=correct pred=missed
  2can-2h-2 LH53 D2 gold=correct pred=wrong_pitch
  2cmajor-2h-up RH1 C4 gold=correct pred=missed
  1ode-1h-miss RH2 E4 gold=missed pred=correct
  2ode-1h-wrong RH5 G4 gold=wrong_pitch pred=correct
  2ode-1h-wrong RH6 F4 gold=wrong_pitch pred=correct
  2ode-1h-wrong RH7 E4 gold=wrong_pitch pred=correct
  2ode-1h-wrong RH8 D4 gold=wrong_pitch pred=missed
  2ode-1h-wrong RH9 C4 gold=correct+late pred=correct
  1emajor-1h-down RH1 E5 gold=correct pred=missed
  1emajor-2h-up RH3 G#4 gold=correct pred=missed
  2emajor-2h-up RH4 A4 gold=correct pred=missed
```
