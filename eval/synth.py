"""
synth.py
Synthetic error cases built from CLEAN monophonic takes, so the model can be checked on hundreds
of labelled mistakes without new recordings.

Two ways to make a mistake out of a correct performance:
Conventions (same as the hand labels): a wrong key followed by the right one is "wrong_pitch" on
that note; only a re-strike of a neighbouring note is "extra".
  * wrong score  - keep the audio, change the score it is graded against
                   (sub, shift, miss_ins_step, miss_ins_repeat, extra_score);
  * spliced time - keep the score, cut / repeat / reorder the feature frames at note boundaries
                   (miss_del, extra_dup, hold, gap, stop_early, restart);
  * held key     - keep the score and the timing, let the previous key go on sounding under the
                   next note (overlap).
Note boundaries come from `reference_onsets`, a model-independent dynamic programme over the
per-pitch onset scores of the clean take.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import dataclasses

import numpy as np

from eval.cache import rescore, splice, sustain
from eval.labels import GoldError, TakeLabels
from eval.pieces import PieceSpec, all_pieces

MONO_PIECES = tuple(n for n, s in all_pieces().items() if s.is_mono)
KINDS = ("clean", "sub", "shift", "miss_ins_step", "miss_ins_repeat", "miss_del",
         "extra_score", "extra_dup", "hold", "gap", "overlap", "stop_early", "restart", "restart_wrong")
PRE = 3            # frames before an onset peak that belong to the new note (attack rise)
HELD_FADE_DB = 0.5  # dB per frame (about 20 dB a second) a key that is kept down fades by: as
                    # measured on the two recorded overlaps (3cmajor-1h-overlap6, 3ode-1h-overlap7)


@dataclass
class Case:
    kind: str
    take_id: str
    bundle: object
    spec: PieceSpec
    gold: TakeLabels
    meta: dict = field(default_factory=dict)


def mono_hand(spec: PieceSpec) -> str:
    hands = spec.playing_hands
    if len(hands) != 1 or any(len(ps) != 1 for ps, _ in spec.hands[hands[0]]):
        raise ValueError(f"{spec.name}: synthetic cases need one hand, no rests and no chords")
    return hands[0]


def reference_onsets(bundle, spec: PieceSpec, min_gap: int = 4) -> np.ndarray:
    """Onset frame of every score note of a clean take: the increasing frame sequence that
    maximises the summed onset score of each note's own pitch."""
    notes = spec.notes()
    T = bundle.audio.num_frames
    S = np.vstack([bundle.candidate_onset_scores[n.pitch] for n in notes])
    dp = np.full((len(notes), T), -np.inf)
    bp = np.zeros((len(notes), T), dtype=int)
    dp[0] = S[0]
    for i in range(1, len(notes)):
        best, arg = -np.inf, 0
        for t in range(min_gap, T):
            if dp[i - 1, t - min_gap] > best:
                best, arg = dp[i - 1, t - min_gap], t - min_gap
            dp[i, t] = best + S[i, t]
            bp[i, t] = arg
    out = np.zeros(len(notes), dtype=int)
    out[-1] = int(np.argmax(dp[-1]))
    for i in range(len(notes) - 1, 0, -1):
        out[i - 1] = bp[i, out[i]]
    return out


def _respec(spec: PieceSpec, hand: str, events) -> PieceSpec:
    """events: (midi pitch or a pitch tuple, beats) -> the piece format, (pitches, beats)."""
    as_chord = lambda p: tuple(p) if isinstance(p, tuple) else ((p,) if p > 0 else ())
    return dataclasses.replace(spec, name=spec.name, hands={hand: [(as_chord(p), d) for p, d in events]})


def _case(kind, take_id, bundle, spec, errors, **meta) -> Case:
    b = rescore(bundle, spec.to_score_model())
    gold = TakeLabels(spec.name, perfect=not errors, errors=errors)
    if "restart" in meta:
        gold.restarts = [meta["restart"]]
    gold.earlier = list(meta.get("earlier", []))
    return Case(kind, take_id, b, spec, gold, meta)


def generate(take_id: str, bundle, spec: PieceSpec, kinds=KINDS):
    """Yield Cases for one clean monophonic take."""
    hand = mono_hand(spec)
    ev = spec.hands[hand]
    P = [ps[0] for ps, _ in ev]
    D = [d for _, d in ev]
    n = len(P)
    nid = lambda i: f"{hand}{i + 1}"
    T = bundle.audio.num_frames
    on = reference_onsets(bundle, spec)
    cut = np.maximum(0, on - PRE)                  # block i = frames [cut[i], cut[i+1])
    cut[0] = 0
    ends = np.append(cut[1:], T)
    block = lambda i: np.arange(cut[i], ends[i])
    ioi = float(np.median(np.diff(on))) if n > 1 else 10.0

    if "clean" in kinds:
        yield _case("clean", take_id, bundle, spec, [])

    if "sub" in kinds:                             # score expects P[i]+d, player played P[i]
        for i in range(n):
            for d in (-2, -1, 1, 2):
                q = list(ev)
                q[i] = (P[i] + d, D[i])
                yield _case("sub", take_id, bundle, _respec(spec, hand, q),
                            [GoldError("wrong_pitch", note_id=nid(i), played=P[i])], note=i, delta=d)

    if "shift" in kinds:                           # hand one position off for four notes
        for i in range(1, n - 4):
            q = list(ev)
            errs = []
            for k in range(i, i + 4):
                q[k] = (P[k + 1], D[k])            # expected = what the player plays one note later
                if P[k + 1] != P[k]:
                    errs.append(GoldError("wrong_pitch", note_id=nid(k), played=P[k]))
            if len(errs) >= 3:
                yield _case("shift", take_id, bundle, _respec(spec, hand, q), errs, start=i)

    for kind in ("miss_ins_step", "miss_ins_repeat"):   # score has a note that was never played
        if kind not in kinds:
            continue
        for i in range(1, n):
            if kind == "miss_ins_repeat":
                q_pitch = P[i - 1]
            else:
                q_pitch = P[i - 1] + (1 if P[i] != P[i - 1] + 1 else 3)
                if q_pitch in (P[i - 1], P[i]):
                    continue
            q = ev[:i] + [(q_pitch, 1.0)] + ev[i:]
            yield _case(kind, take_id, bundle, _respec(spec, hand, q),
                        [GoldError("missed", note_id=nid(i))], note=i)

    if "miss_del" in kinds:                        # the note's frames are cut out of the recording
        for i in range(1, n - 1):
            frames = np.concatenate([np.arange(0, cut[i]), np.arange(ends[i], T)])
            yield _case("miss_del", take_id, splice(bundle, frames), spec,
                        [GoldError("missed", note_id=nid(i))], note=i)

    if "extra_score" in kinds:                     # a played note has no place in the score
        for i in range(1, n - 1):
            q = ev[:i] + ev[i + 1:]
            if P[i] in (P[i - 1], P[i + 1]):       # re-strike of a neighbour: an extra note
                gold = [GoldError("extra", after=nid(i - 1), played=P[i])]
            else:                                  # a different key before the right one: labelled, as in the
                gold = [GoldError("wrong_pitch", note_id=nid(i), played=P[i])]   # real takes, as wrong pitch
            yield _case("extra_score", take_id, bundle, _respec(spec, hand, q), gold, note=i)

    if "extra_dup" in kinds:                       # a note is struck twice
        for i in range(1, n - 1):
            frames = np.concatenate([np.arange(0, ends[i]), block(i), np.arange(ends[i], T)])
            yield _case("extra_dup", take_id, splice(bundle, frames), spec,
                        [GoldError("extra", after=nid(i), played=P[i])], note=i)

    # a gap of twice the written length before note i ...
    pause = int(round(1.0 * ioi))
    for kind in ("hold", "gap"):
        if kind not in kinds:
            continue
        for i in range(1, n):
            if kind == "hold":                     # ... filled by the previous note ringing on: the next
                src = np.arange(max(cut[i - 1] + PRE + 2, cut[i] - 3), cut[i])     # note still STARTS late
                gold = [GoldError("late", note_id=nid(i))]
            else:                                  # ... silent: the next note comes late
                src = np.array([-1])                # silent frames
                gold = [GoldError("late", note_id=nid(i))]
            if src.size == 0:
                continue
            frames = np.concatenate([np.arange(0, cut[i]), np.resize(src, pause), np.arange(cut[i], T)])
            yield _case(kind, take_id, splice(bundle, frames), spec, gold, note=i, pause_frames=pause)

    if "overlap" in kinds:                         # note i-1 is not released: its key sounds on under
        for i in range(1, n):                      # note i, which starts on time -> held_long on note i-1
            src = min(on[i - 1] + 4, cut[i] - 2)   # note i-1 in full sound: after its attack, before the next
            if P[i] == P[i - 1] or src <= on[i - 1]:
                continue
            frames = np.arange(src + 1, ends[i])   # it fades from the next strike on, to the end of note i
            fade = HELD_FADE_DB * np.maximum(frames - on[i], 0)
            yield _case("overlap", take_id, sustain(bundle, P[i - 1], src, frames, fade),
                        spec, [GoldError("held_long", note_id=nid(i - 1))], note=i)

    if "stop_early" in kinds:                      # player stops; the rest is never played
        for j in sorted({n // 2, n - 2}):
            if j < 2:
                continue
            yield _case("stop_early", take_id, splice(bundle, np.arange(0, cut[j])), spec,
                        [GoldError("missed", note_id=nid(k)) for k in range(j, n)], stop=j)

    if "restart" in kinds:                         # plays up to note j, goes back to r, plays to the end
        notes = spec.notes()
        bar_starts = [k for k, nt in enumerate(notes) if abs(nt.beat_in_measure - 1.0) < 1e-6]
        for j in range(3, n):
            # going back a single note is a stutter (extra_dup), not a restart
            targets = {r for r in (0, max([b for b in bar_starts if b < j], default=0)) if j - r >= 2}
            for r in sorted(targets):
                frames = np.concatenate([np.arange(0, cut[j]), np.arange(cut[r], T)])
                yield _case("restart", take_id, splice(bundle, frames), spec, [],
                            stop=j, resume=r, restart=(nid(j - 1), nid(r)))

    if "restart_wrong" in kinds:                   # as restart, but the last note before going back was a
        notes = spec.notes()                       # wrong key (the sound of a neighbouring note of another pitch)
        bar_starts = [k for k, nt in enumerate(notes) if abs(nt.beat_in_measure - 1.0) < 1e-6]
        for j in range(3, n):
            i = j - 1
            length = int(cut[j] - cut[i])
            targets = {r for r in (0, max([b for b in bar_starts if b < j], default=0)) if j - r >= 2}
            for r in sorted(targets):
                # the wrong key must not equal the note before the resume point or the note before the
                # stop: then "went back one note further" or "struck that key again" is the same sound
                avoid = {P[i], P[i - 1]} | ({P[r - 1]} if r > 0 else set())
                k = next((x for x in (j, i - 2, i + 1, i - 3, j + 1) if 0 <= x < n and P[x] not in avoid), None)
                if k is None or length <= 0:
                    continue
                src = np.arange(cut[k], cut[k + 1] if k + 1 < n else T)
                if src.size == 0:
                    continue
                wrong = np.concatenate([src, np.full(max(0, length - src.size), src[-1])])[:length]
                frames = np.concatenate([np.arange(0, cut[i]), wrong, np.arange(cut[r], T)])
                yield _case("restart_wrong", take_id, splice(bundle, frames), spec, [],
                            stop=j, resume=r, restart=(nid(i), nid(r)),
                            earlier=[GoldError("wrong_pitch", note_id=nid(i), played=P[k], attempt=1)])
