"""
stft_hmm_v3.py  --  Note-level HMM over detected key strikes.

v2 decoded frames and then asked "is the expected pitch present in this slice?". v3 first finds
what was struck and when (strike events), then aligns the event sequence to the score with a
hidden Markov model whose hidden state is the score position:

    event matched to the next position, pitch agrees      -> correct
    event matched to a position, pitch differs            -> wrong pitch (and which pitch)
    position passed over without an event                 -> missed
    event matched to no position                          -> extra
    event matched to an earlier position                  -> the player went back (restart)

The time between matched events is compared with the score's beat distance at the local tempo
(explicit-duration term), which keeps a run of wrong notes from sliding the alignment and gives
the late / held-long verdicts. Features come unchanged from stft_features.py.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, field

import numpy as np

import stft_features
from score_models import midi_to_note_name

OVERTONE_OFFSETS = (12, 19, 24, 28, 31, 36)


@dataclass
class V3Config:
    # --- strike detection
    onset_floor: float = 0.40         # per-pitch onset score below this is never a strike ...
    onset_full: float = 0.65          # ... and at or above this it counts fully (real strikes measure >= 0.65)
    presence_window: int = 7          # salience = median presence over this many frames after the onset
    min_frames_after: int = 5         # too close to the end of the recording to judge
    min_salience: float = 0.15        # weaker candidates are dropped outright
    cluster_gap: int = 3              # strikes this close in frames belong to one event
    same_pitch_gap: int = 7           # two onset peaks of one pitch closer than this are one strike
    overtone_ratio: float = 1.10      # a harmonic no stronger than this x a lower note is its overtone
    ghost_ratio: float = 0.85         # a sub-harmonic weaker than this x the note above it is a ghost
    leak_ratio: float = 0.80          # a semitone neighbour weaker than this x its neighbour is leakage
    match_min_salience: float = 0.30  # an expected pitch counts as struck from this salience up
    # score pitches that sit on a harmonic of another score pitch (octave, fifth, third above the
    # bass) are erased by the front end's overtone subtraction; for those only, a strike is also
    # read straight off the spectrum: the fundamental's level must jump and stand out as a peak
    raw_rise_db: float = 6.0
    raw_peak_db: float = 10.0
    # --- alignment costs (lower is better)
    sub_cost: float = 3.0             # event explains a position with the wrong pitch
    sub_cost_near: float = 2.5        # ... when it is a neighbouring key (within a whole tone): the usual slip
    skip_time_slack: float = 1.0      # a dropped note usually drops its time too: allow that reading at this cost
    extra_time_slack: float = 1.5     # an added note usually adds its time: likewise
    sub_min_salience: float = 0.40    # a weak event cannot stand in for a note
    skip_cost: float = 4.0            # a score note with no event
    jump_cost: float = 9.0            # a long forward skip costs jump_cost + jump_step per note at most
    jump_step: float = 0.25
    extra_cost: float = 4.0           # a clear event that belongs to no score note
    extra_cost_weak: float = 0.5      # the same for a faint event (likely noise)
    extra_weak_salience: float = 0.30
    extra_strong_salience: float = 0.45
    max_extras: int = 6               # consecutive unmatched events allowed
    timing_weight: float = 6.25       # x (ln of observed/expected inter-onset time)^2 ; 2x off -> 3.0
    timing_cap: float = 4.0
    chord_missing_cost: float = 2.0   # per expected pitch absent from a matched event (two voices)
    stop_cost: float = 6.0            # at most this much for the notes left unplayed at the end
    back_cost_bar: float = 5.0        # going back to the start of a bar (or of the piece) and playing on
    back_cost_any: float = 7.0        # going back to any other earlier note
    back_min_notes: int = 2           # replaying fewer notes than this is a re-strike (an extra), not a restart
    # --- timing verdicts
    slow_ratio: float = 1.5           # a gap this many times its written length is a timing error:
    sounding_level: float = 0.5       #   note still sounding (>= this x its level) up to the next strike
    sounding_frames: int = 5          #   over these frames before it -> "held too long"; silent -> next note "late"
    # --- reporting
    smudge_ratio: float = 0.90        # a stray key struck this strongly next to the right one: not a clean note
    extra_report_salience: float = 0.40
    tempo_passes: int = 2


@dataclass
class Event:
    frame: int
    strikes: dict[int, float]                       # every candidate pitch -> salience
    primary: list[tuple[int, float]]                # after removing overtones / leakage / ghosts, strongest first
    explained_by: dict[int, int] = field(default_factory=dict)   # removed pitch -> pitch that explains it
    raw_only: set = field(default_factory=set)       # pitches seen only as a jump in the spectrum

    @property
    def salience(self) -> float:
        return self.primary[0][1] if self.primary else 0.0

    @property
    def pitch(self) -> int | None:
        return self.primary[0][0] if self.primary else None


@dataclass(frozen=True)
class Position:
    index: int
    beat: float
    pitches: tuple[int, ...]
    segment_index: int
    bar_start: bool = False


@dataclass
class Match:
    event: int
    position: int
    kind: str                # 'next' | 'skip' | 'back'


@dataclass
class Analysis:
    events: list[Event]
    positions: list[Position]
    matches: list[Match]
    cost: float
    seconds_per_beat: np.ndarray        # local tempo at each event
    seconds_per_frame: float


# ---------------------------------------------------------------- strike events
def _peaks(x: np.ndarray, height: float, gap: int) -> list[int]:
    out, taken = [], np.zeros(x.size, dtype=bool)
    for i in np.argsort(x)[::-1]:
        if x[i] < height:
            break
        if taken[i]:
            continue
        out.append(int(i))
        taken[max(0, i - gap + 1):i + gap] = True
    return sorted(out)


def harmonic_score_pitches(bundle) -> list[int]:
    pitches = sorted({p for seg in bundle.segments for p in seg.expected_pitches})
    return [q for q in pitches if any(q - p in OVERTONE_OFFSETS for p in pitches)]


def raw_strikes(bundle, pitch: int, cfg: V3Config) -> list[int]:
    """Frames where the fundamental of `pitch` jumps by raw_rise_db and then stands raw_peak_db
    above the neighbouring bins. Says that energy arrived at that frequency, not which key sent it."""
    db = bundle.audio.stft_db
    n_bins, T = db.shape
    k = int(round(stft_features.midi_to_frequency(pitch) * (n_bins - 1) * 2 / bundle.audio.sample_rate))
    if k < 9 or k + 9 >= n_bins or T < 12:
        return []
    level = db[k - 1:k + 2].max(axis=0).astype(np.float64)
    around = np.median(np.vstack([db[k - 8:k - 3], db[k + 4:k + 9]]).astype(np.float64), axis=0)

    def window_mean(x, first, last):                   # mean of x[t+first .. t+last], edges repeated
        xp = np.pad(x, 8, mode="edge")
        return np.mean([xp[8 + d:8 + d + T] for d in range(first, last + 1)], axis=0)

    after = window_mean(level, 1, 3)
    rise = after - window_mean(level, -4, -2)
    rise[(after - window_mean(around, 1, 3)) < cfg.raw_peak_db] = 0.0
    return [f for f in _peaks(rise, cfg.raw_rise_db, cfg.same_pitch_gap) if f + cfg.min_frames_after <= T]


def detect_events(bundle, cfg: V3Config) -> list[Event]:
    T = bundle.audio.num_frames
    cands: list[tuple[int, int, float]] = []          # (frame, pitch, salience)
    raw: set[tuple[int, int]] = set()
    for q in harmonic_score_pitches(bundle):
        for f in raw_strikes(bundle, q, cfg):
            cands.append((f, q, cfg.match_min_salience))
            raw.add((f, q))
    for p, onset in bundle.candidate_onset_scores.items():
        pres = bundle.candidate_scores[p]
        for f in _peaks(onset, cfg.onset_floor, cfg.same_pitch_gap):
            if f + cfg.min_frames_after > T:
                continue
            gain = min(1.0, (float(onset[f]) - cfg.onset_floor) / (cfg.onset_full - cfg.onset_floor))
            s = gain * float(np.median(pres[f + 1:min(T, f + 1 + cfg.presence_window)]))
            if s >= cfg.min_salience:
                cands.append((f, p, s))
    cands.sort()
    groups: list[list[tuple[int, int, float]]] = []
    for c in cands:
        if groups and c[0] - groups[-1][-1][0] <= cfg.cluster_gap:
            groups[-1].append(c)
        else:
            groups.append([c])
    events = []
    for g in groups:
        strikes: dict[int, float] = {}
        frames: dict[int, int] = {}
        confirmed = {p for f, p, s in g if (f, p) not in raw}
        for f, p, s in g:
            if s > strikes.get(p, 0.0):
                strikes[p], frames[p] = s, f
        explained: dict[int, int] = {}
        kept: list[int] = []
        # strongest first: the strongest pitch is real; weaker ones related to a kept pitch are its
        # overtone (above), its sub-harmonic ghost (below, clearly weaker) or semitone leakage
        for q in sorted(strikes, key=lambda k: -strikes[k]):
            why = None
            for k in kept:
                if q - k in OVERTONE_OFFSETS:
                    why = k
                elif k - q in OVERTONE_OFFSETS and strikes[q] < cfg.ghost_ratio * strikes[k]:
                    why = k
                elif abs(q - k) == 1 and strikes[q] < cfg.leak_ratio * strikes[k]:
                    why = k
                if why is not None:
                    break
            if why is None:
                kept.append(q)
            else:
                explained[q] = why
        # a real low note can measure slightly weaker than its own octave: fold the octave back in
        for q in sorted(kept):
            src = next((q - off for off in OVERTONE_OFFSETS
                        if q - off in kept and strikes[q] <= cfg.overtone_ratio * strikes[q - off]), None)
            if src is not None:
                kept.remove(q)
                explained[q] = src
        primary = sorted(((q, strikes[q]) for q in kept), key=lambda x: -x[1])
        events.append(Event(frames[primary[0][0]], strikes, primary, explained, set(strikes) - confirmed))
    return events


def score_positions(bundle, beats_per_measure: float = 4.0, pickup_beats: float = 0.0) -> list[Position]:
    out = []
    for seg in bundle.segments:
        if seg.onset_pitches:
            in_bar = (seg.start_beat - pickup_beats) % beats_per_measure
            out.append(Position(len(out), seg.start_beat, tuple(seg.onset_pitches), seg.index,
                                bar_start=min(in_bar, beats_per_measure - in_bar) < 1e-6 or not out))
    return out


def back_costs(positions: list[Position], cfg: V3Config) -> np.ndarray:
    """[j', j] cost of resuming at an earlier position j after playing j' (inf where not allowed).
    Learners restart where the music restarts: at a bar line far more often than mid-bar."""
    J = len(positions)
    jj = np.arange(J)
    target = np.where([p.bar_start for p in positions], cfg.back_cost_bar, cfg.back_cost_any)
    cost = np.broadcast_to(target[None, :], (J, J)).copy()
    cost[jj[:, None] - jj[None, :] < cfg.back_min_notes - 1] = np.inf    # replaying j..j' is j'-j+1 notes
    return cost


# ---------------------------------------------------------------- emission
def struck(event: Event, pitch: int, expected: tuple[int, ...], cfg: V3Config) -> float:
    """Salience with which `pitch` was struck in `event`, 0 if it was not. A pitch removed as an
    overtone still counts when the note that explains it is itself expected here (octaves, fifths)."""
    s = event.strikes.get(pitch, 0.0)
    if s < cfg.match_min_salience:
        return 0.0
    src, hops = event.explained_by.get(pitch), 0
    while src is not None and src in event.explained_by and hops < 8:
        src, hops = event.explained_by[src], hops + 1
    if src is None or src in expected:
        return s
    return 0.0


def emission_cost(event: Event, pos: Position, cfg: V3Config) -> float:
    missing = [p for p in pos.pitches if struck(event, p, pos.pitches, cfg) <= 0.0]
    if not missing:
        return 0.0
    if len(missing) < len(pos.pitches):
        return cfg.chord_missing_cost * len(missing)
    if event.salience < cfg.sub_min_salience:
        return np.inf
    near = min(abs(event.pitch - p) for p in pos.pitches) <= 2
    return (cfg.sub_cost_near if near else cfg.sub_cost) + cfg.chord_missing_cost * (len(missing) - 1)


def extra_cost(event: Event, cfg: V3Config) -> float:
    w = np.clip((event.salience - cfg.extra_weak_salience)
                / (cfg.extra_strong_salience - cfg.extra_weak_salience), 0.0, 1.0)
    return float(cfg.extra_cost_weak + w * (cfg.extra_cost - cfg.extra_cost_weak))


# ---------------------------------------------------------------- alignment (Viterbi over events)
def _timing(ioi_s: float, beats: np.ndarray, spb: float, cfg: V3Config) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(max(ioi_s, 1e-3) / (beats * spb))
    return np.minimum(cfg.timing_weight * r * r, cfg.timing_cap)


def align(events: list[Event], positions: list[Position], cfg: V3Config,
          spb: np.ndarray, spf: float, back_cost=None) -> tuple[list[Match], float]:
    """Cheapest explanation of the events by the score. dp[e, j] = best cost of a path whose last
    matched pair is (event e, position j)."""
    E, J = len(events), len(positions)
    if E == 0 or J == 0:
        return [], float(min(J * cfg.skip_cost, cfg.stop_cost) if J else 0.0)
    beat = np.array([p.beat for p in positions])
    t = np.array([ev.frame for ev in events]) * spf
    emit = np.array([[emission_cost(ev, pos, cfg) for pos in positions] for ev in events])
    xc = np.array([extra_cost(ev, cfg) for ev in events])
    xcum = np.concatenate([[0.0], np.cumsum(xc)])                 # xcum[e] = cost of events 0..e-1 all extra
    jj = np.arange(J)
    skips = jj[None, :] - jj[:, None] - 1                          # [j', j] notes passed over
    skip_c = np.where(skips >= 0, np.minimum(skips * cfg.skip_cost, cfg.jump_cost + skips * cfg.jump_step), np.inf)
    dbeat = beat[None, :] - beat[:, None]                          # [j', j]
    own = np.broadcast_to(np.diff(beat, append=beat[-1] + 1.0)[:, None], (J, J))    # beats of note j' itself
    fwd = skips >= 0
    skipped = skips >= 1

    dp = np.full((E, J), np.inf)
    bp = np.full((E, J, 2), -1, dtype=int)
    start = np.minimum(jj * cfg.skip_cost, cfg.jump_cost + jj * cfg.jump_step)
    for e in range(E):
        dp[e] = xcum[e] + start + emit[e]
        for e0 in range(max(0, e - 1 - cfg.max_extras), e):
            between = xcum[e] - xcum[e0 + 1]
            tr = np.full((J, J), np.inf)
            ioi = t[e] - t[e0]
            tr[fwd] = skip_c[fwd] + _timing(ioi, dbeat[fwd], spb[e], cfg)        # time of the skipped notes kept
            tr[skipped] = np.minimum(tr[skipped], skip_c[skipped] + cfg.skip_time_slack
                                     + _timing(ioi, own[skipped], spb[e], cfg))   # ... or dropped with them
            if e0 < e - 1:                       # extras in between: they may have added their own time
                tr[fwd] = np.minimum(tr[fwd], skip_c[fwd] + cfg.extra_time_slack
                                     + _timing(t[e] - t[e - 1], dbeat[fwd], spb[e], cfg))
            if back_cost is not None:
                tr = np.minimum(tr, back_cost)
            cand = dp[e0][:, None] + tr + between                  # [j', j]
            j0 = np.argmin(cand, axis=0)
            best = cand[j0, jj] + emit[e]
            better = best < dp[e]
            dp[e][better] = best[better]
            bp[e][better, 0] = e0
            bp[e][better, 1] = j0[better]
    left = J - 1 - jj
    end = np.minimum(left * cfg.skip_cost, cfg.stop_cost + left * cfg.jump_step)
    total = dp + (xcum[E] - xcum[1:E + 1])[:, None] + end[None, :]
    e, j = np.unravel_index(int(np.argmin(total)), total.shape)
    cost = float(total[e, j])
    if not np.isfinite(cost):
        return [], float(xcum[E] + min(J * cfg.skip_cost, cfg.stop_cost))
    path = []
    while e >= 0:
        e0, j0 = bp[e, j]
        kind = "next" if e0 < 0 or j == j0 + 1 else ("back" if j <= j0 else "skip")
        path.append(Match(int(e), int(j), kind))
        e, j = int(e0), int(j0)
    return path[::-1], cost


def _initial_spb(events: list[Event], positions: list[Position], spf: float) -> float:
    strong = [ev.frame for ev in events if ev.salience >= 0.45]
    steps = np.diff([p.beat for p in positions])
    if len(strong) < 3 or steps.size == 0:
        return 0.5
    return float(np.median(np.diff(strong)) * spf / np.median(steps))


def _local_spb(events, positions, matches, spf, fallback: float) -> np.ndarray:
    """Tempo at each event from neighbouring cleanly matched pairs (running median)."""
    ts, vals = [], []
    for a, b in zip(matches, matches[1:]):
        if b.kind == "next" and b.position == a.position + 1 and all(
                events[k].salience < 0.4 for k in range(a.event + 1, b.event)):
            beats = positions[b.position].beat - positions[a.position].beat
            ioi = (events[b.event].frame - events[a.event].frame) * spf
            if beats > 0 and ioi > 0:
                ts.append(events[b.event].frame)
                vals.append(ioi / beats)
    E = len(events)
    if len(vals) < 3:
        return np.full(E, fallback)
    vals = np.array(vals)
    g = float(np.median(vals))
    sm = np.array([np.median(vals[max(0, i - 3):i + 4]) for i in range(len(vals))])
    sm = np.clip(sm, 0.6 * g, 1.6 * g)
    return np.interp([ev.frame for ev in events], ts, sm)


def analyze(bundle, cfg: V3Config | None = None, beats_per_measure: float = 4.0,
            pickup_beats: float = 0.0, backtracking: bool = True) -> Analysis:
    cfg = cfg or V3Config()
    spf = bundle.audio.seconds_per_frame
    events = detect_events(bundle, cfg)
    positions = score_positions(bundle, beats_per_measure, pickup_beats)
    E = len(events)
    back = back_costs(positions, cfg) if backtracking and positions else None
    base = _initial_spb(events, positions, spf)
    best = None
    for mult in (0.5, 0.71, 1.0, 1.41, 2.0):
        spb = np.full(E, base * mult)
        m, c = align(events, positions, cfg, spb, spf, back)
        if best is None or c < best[1]:
            best = (m, c, spb)
    matches, cost, spb = best
    for _ in range(cfg.tempo_passes):                 # follow tempo drift: re-estimate locally, re-align
        spb = _local_spb(events, positions, matches, spf, float(np.median(spb)) if E else base)
        matches, cost = align(events, positions, cfg, spb, spf, back)
    return Analysis(events, positions, matches, cost, spb, spf)


# ---------------------------------------------------------------- verdicts
@dataclass
class PitchVerdict:
    pitch: int
    label: str                     # correct | wrong_pitch | missed
    played: int | None = None
    frame: int | None = None
    corrected: bool = False        # a wrong key first, then the right one
    flags: set = field(default_factory=set)      # held_long | late


@dataclass
class Report:
    verdicts: dict[int, list[PitchVerdict]]            # position index -> one verdict per expected pitch
    extras: list[tuple[int | None, int, int]]          # (position index it follows or None, pitch, frame)
    restarts: list[tuple[int, int]]                    # (position before the jump, position resumed at)
    visits: dict[int, list[int]]                       # position index -> events matched to it, in order


def verdicts_for(event: Event, pos: Position, prev_pitches: tuple[int, ...], cfg: V3Config) -> tuple[list[PitchVerdict], list[int]]:
    """Per-pitch verdicts for one matched (event, position), plus leftover struck pitches."""
    out, used = [], set()
    free = [(q, s) for q, s in event.primary if q not in pos.pitches]
    for p in pos.pitches:
        s = struck(event, p, pos.pitches, cfg)
        if s > 0.0:
            # two keys at once with the stray one at least as strong: the note was not played cleanly
            stray = [(q, sq) for q, sq in free if q not in used and q not in prev_pitches and sq >= cfg.smudge_ratio * s]
            # an expected pitch seen only as a jump in the spectrum loses to a clearly struck other key
            doubtful = p in event.raw_only and any(sq >= cfg.sub_min_salience for _, sq in stray)
            stray = [x for x in stray if x[1] >= cfg.sub_min_salience] if doubtful else stray
            if (len(pos.pitches) == 1 or doubtful) and stray:
                used.add(stray[0][0])
                out.append(PitchVerdict(p, "wrong_pitch", stray[0][0], event.frame))
            else:
                out.append(PitchVerdict(p, "correct", p, event.frame))
            continue
        cand = [(abs(q - p), q) for q, sq in free if q not in used and sq >= cfg.sub_min_salience]
        if cand:
            q = min(cand)[1]
            used.add(q)
            out.append(PitchVerdict(p, "wrong_pitch", q, event.frame))
        else:
            out.append(PitchVerdict(p, "missed", None, event.frame))
    return out, [q for q, s in free if q not in used]


def still_sounding(bundle, pitch: int, onset: int, until: int, cfg: V3Config) -> bool:
    """Is `pitch`, struck at frame `onset`, still sounding just before frame `until`?"""
    pres = bundle.candidate_scores[pitch]
    level = float(np.median(pres[onset + 2:onset + 8])) if onset + 2 < len(pres) else 0.0
    lo, hi = max(onset + 2, until - 1 - cfg.sounding_frames), until - 1
    if hi <= lo or level <= 0.0:
        return True
    return float(np.mean(pres[lo:hi] >= cfg.sounding_level * level)) >= 0.5


def timing_flags(an: Analysis, rep: "Report", bundle, cfg: V3Config) -> None:
    """A gap between two consecutive score notes that lasts >= slow_ratio x its written length:
    if the first note rang on through the gap it was held too long, otherwise the second came late."""
    spf = an.seconds_per_frame
    for a, c in zip(an.matches, an.matches[1:]):
        if c.position != a.position + 1 or any(
                an.events[k].salience >= cfg.extra_report_salience for k in range(a.event + 1, c.event)):
            continue                                   # skips, restarts and extra notes explain their own time
        va, vc = rep.verdicts[a.position], rep.verdicts[c.position]
        if any(v.corrected for v in vc):
            continue
        fa, fc = an.events[a.event].frame, an.events[c.event].frame
        beats = an.positions[c.position].beat - an.positions[a.position].beat
        ratio = (fc - fa) * spf / (beats * an.seconds_per_beat[c.event])
        if ratio < cfg.slow_ratio:
            continue
        sounding = [v for v in va if v.played is not None and still_sounding(bundle, v.played, fa, fc, cfg)]
        if sounding:
            for v in sounding:
                v.flags.add("held_long")
        else:
            for v in vc:
                v.flags.add("late")


def report(an: Analysis, cfg: V3Config | None = None, bundle=None) -> Report:
    """Turn the alignment into verdicts.

    An unmatched event is one of three things: a re-strike of the note just played or a double
    strike of the note that follows (an extra note); a wrong key pressed before the right one
    (the following note is reported as wrong pitch, corrected); or noise (too faint to report)."""
    cfg = cfg or V3Config()
    verdicts: dict[int, list[PitchVerdict]] = {}
    visits: dict[int, list[int]] = {}
    extras, restarts = [], []
    matched = {m.event: m for m in an.matches}
    last_pos, prev_pitches = None, ()
    pending: list[Event] = []                          # clear unmatched events since the last match
    for e, ev in enumerate(an.events):
        m = matched.get(e)
        if m is None:
            if ev.salience >= cfg.extra_report_salience:
                pending.append(ev)
            continue
        pos = an.positions[m.position]
        if m.kind == "back" and last_pos is not None:
            restarts.append((last_pos, m.position))
        vs, _ = verdicts_for(ev, pos, prev_pitches, cfg)
        prev_set = set(an.positions[last_pos].pitches) if last_pos is not None else set()
        for x in pending:
            target = min(vs, key=lambda v: abs(v.pitch - x.pitch))
            if x.pitch in prev_set or x.pitch in pos.pitches or m.kind == "back" or target.label != "correct":
                extras.append((last_pos, x.pitch, x.frame))
            else:
                target.label, target.played, target.corrected = "wrong_pitch", x.pitch, True
        pending = []
        verdicts[m.position] = vs                                          # the last visit decides
        visits.setdefault(m.position, []).append(e)
        last_pos = m.position
        prev_pitches = tuple(v.played for v in vs if v.played is not None)
    extras += [(last_pos, x.pitch, x.frame) for x in pending]
    for pos in an.positions:
        if pos.index not in verdicts:
            verdicts[pos.index] = [PitchVerdict(p, "missed") for p in pos.pitches]
    rep = Report(verdicts, extras, restarts, visits)
    if bundle is not None:
        timing_flags(an, rep, bundle, cfg)
    return rep


def print_report(an: Analysis, rep: Report) -> None:
    spf = an.seconds_per_frame
    print("=" * 78)
    print("v3 NOTE-LEVEL HMM REPORT")
    print("=" * 78)
    for pos in an.positions:
        for v in rep.verdicts[pos.index]:
            when = f"{v.frame * spf:6.2f}s" if v.frame is not None else "   -   "
            what = {"correct": "correct", "missed": "MISSED",
                    "wrong_pitch": f"WRONG NOTE (played {midi_to_note_name(v.played or 0)}"
                                   f"{', then corrected' if v.corrected else ''})"}[v.label]
            again = f"  [played {len(rep.visits[pos.index])}x]" if len(rep.visits.get(pos.index, [])) > 1 else ""
            again += "".join({"held_long": "  HELD TOO LONG", "late": "  LATE"}[f] for f in sorted(v.flags))
            print(f"  note {pos.index + 1:3d}  beat {pos.beat:5.2f}  {midi_to_note_name(v.pitch):<4} {when}  {what}{again}")
    for after, pitch, frame in rep.extras:
        where = f"after note {after + 1}" if after is not None else "before the first note"
        print(f"  EXTRA {midi_to_note_name(pitch)} at {frame * spf:.2f}s ({where})")
    for a, b in rep.restarts:
        print(f"  WENT BACK from note {a + 1} to note {b + 1}")
    print("=" * 78)


def run_pipeline(audio_path, model_name):
    cfg = V3Config()
    bundle = stft_features.run_stage1_and_stage2(audio_path, model_name)
    an = analyze(bundle, cfg)
    rep = report(an, cfg, bundle)
    print_report(an, rep)
    return bundle, an, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", "--mp3", dest="audio", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    run_pipeline(a.audio, a.model)
