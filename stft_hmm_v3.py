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
the late verdict. Features come unchanged from stft_features.py.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, field

import numpy as np

import stft_features
from score_models import midi_to_note_name

OVERTONE_OFFSETS = (12, 19, 24, 28, 31, 36)
PHANTOM_OFFSETS = (5, 7, 12, 19)      # fourth, fifth, octave, octave + fifth: keys that share a low partial


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
    # strike strength is judged against what a clear strike of that key measures in this recording
    # (keys differ: on one piano C4 reads ~0.76 and E5 ~0.40; so do pianos, phones and players).
    # Every salience threshold in this file is written for a key whose clear strikes read level_reference.
    relative_levels: bool = True
    strike_needs_sound: bool = True   # nothing is struck where the recording is at its quiet floor
    strike_energy: float = 0.01       # ... frame energy below this share of the recording's loudness range
    level_reference: float = 0.65     # the level the thresholds were written for
    level_span: int = 2               # a key's level also draws on keys within this many semitones ...
    level_prior: float = 2.0          # ... which weigh as much as this many strikes of the key itself
    level_scale_min: float = 1.0      # quiet keys are raised (up to level_scale_max); a key is never marked down for loud neighbours
    level_scale_max: float = 2.5
    # a key struck again while it is still ringing barely moves its onset score (the pitch is already
    # there). It is accepted from a lower onset score when the key's own level jumps and the overall
    # loudness rises with it, and then counts at full strength.
    restrike_floor: float = 0.20      # onset score from which a re-strike of a ringing key is considered
    restrike_presence: float = 0.20   # the key must be sounding before and after
    restrike_rise_db: float = 3.0     # jump of the key's fundamental
    restrike_energy: float = 0.15     # rise of the frame energy (0..1 scale)
    # score pitches that sit on a harmonic of another score pitch (octave, fifth, third above the
    # bass) are erased by the front end's overtone subtraction; for those only, a strike is also
    # read straight off the spectrum: the fundamental's level must jump and stand out as a peak
    raw_rise_db: float = 6.0
    raw_peak_db: float = 10.0
    hand_span: int = 12               # one hand reaches about an octave: a key further off was struck by the other hand
    # --- alignment costs (lower is better)
    sub_cost: float = 3.0             # event explains a position with the wrong pitch
    sub_cost_near: float = 2.5        # ... when it is a neighbouring key (within a whole tone): the usual slip
    skip_time_slack: float = 1.0      # a dropped note usually drops its time too: allow that reading at this cost
    extra_time_slack: float = 1.5     # an added note usually adds its time: likewise
    sub_min_salience: float = 0.40    # an event at least this clear stands in for a note at the plain cost;
    sub_weak_penalty: float = 3.0     # a fainter one (down to match_min_salience) pays up to this much more
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
    late_beats: float = 0.33          # a note that starts a third of a beat after its place is late ...
    late_fraction: float = 0.2        # ... or this fraction of the written gap, whichever is larger (long notes drift more)
    sounding_level: float = 0.5       # reason for a late note: the previous key still sounding (>= this x its
    sounding_frames: int = 5          #   level over these frames before the strike, and the room not silent)
    silence_energy: float = 0.02      #   -> "held"; otherwise "paused"
    late_after_missed: bool = True    # also judge the note that follows a missed note (its time is kept)
    # --- reporting
    smudge_ratio: float = 0.90        # a stray key struck this strongly next to the right one: not a clean note
    extra_report_salience: float = 0.40
    restrike_report_ratio: float = 0.6   # a fainter unmatched strike is still an extra when it repeats the
                                         # neighbouring note's key at this fraction of that note's strength
    correction_max_beats: float = 1.5    # a wrong key is "corrected" by the note that follows within this many
                                         # beats; a stray key further ahead of the note is an extra instead
    tempo_passes: int = 2
    # --- keys that share a partial: a key a fourth, fifth or octave from a struck key collects that key's
    # shared partial in its own score, so it can look struck without sounding (a "phantom": E5 lights up
    # B5, G3 lights up C3). The raw spectrum at each key's own fundamental tells them apart.
    phantom_gap_db: float = 12.0      # a related key whose fundamental is this much quieter than the struck
                                      #   key's is a phantom: never a stray, wrong or extra key, and it stands
                                      #   for an expected note only when its louder partner is expected too
    ring_rise_db: float = 6.0         # a key whose fundamental rose less than this was already ringing: it
                                      #   cannot pass off a freshly struck expected key as its overtone
    # --- search window: the alignment only considers moves a player makes, so its work grows with
    # the length of the piece instead of with its square
    max_skip: int = 16                # a forward jump passes over at most this many notes
    back_bars: int = 4                # a restart goes back at most this many bars, or to the very beginning
    # --- held too long (overlap): the previous key still down under a next note that came on time
    held_after_s: float = 0.25        # judged this long after the next strike (a shorter overlap is ordinary legato)
    held_drop_share: float = 0.33     # held = the key's level fell by less than this share of what the
                                      #   recording's keys usually fall by once the next note is struck


@dataclass
class Event:
    frame: int
    strikes: dict[int, float]                       # every candidate pitch -> salience
    primary: list[tuple[int, float]]                # after removing overtones / leakage / ghosts, strongest first
    explained_by: dict[int, int] = field(default_factory=dict)   # removed pitch -> pitch that explains it
    raw_only: set = field(default_factory=set)       # pitches seen only as a jump in the spectrum
    full: set = field(default_factory=set)           # pitches whose attack counted at full strength
    phantom: dict = field(default_factory=dict)      # phantom key -> the louder related key it borrows from
    rise: dict = field(default_factory=dict)         # pitch -> dB its own fundamental rose at its strike

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
    """Score pitches that share partials with another score pitch: the upper or the lower note of
    an octave, fifth or third (plus octaves) written somewhere in the piece."""
    pitches = sorted({p for seg in bundle.segments for p in seg.expected_pitches})
    return [q for q in pitches if any(abs(q - p) in OVERTONE_OFFSETS for p in pitches)]


def _spectral_jumps(bundle, hz: float, cfg: V3Config):
    """(frames, after, peak) for the frequency `hz`: the frames where its level jumps by raw_rise_db
    and then stands raw_peak_db above the neighbouring bins; the level just after every frame; and
    how far that level stands above the neighbouring bins. ([], None, None) outside the spectrum."""
    db = bundle.audio.stft_db
    n_bins, T = db.shape
    k = int(round(hz * (n_bins - 1) * 2 / bundle.audio.sample_rate))
    if k < 9 or k + 9 >= n_bins or T < 12:
        return [], None, None
    # the bins either side belong to the note too, unless the next semitone is under 3 bins away (low
    # notes): the window's main lobe spans 2 bins either side, so that note would reach into them
    r = 1 if k * (2.0 ** (1.0 / 12.0) - 1.0) >= 3.0 else 0
    level = db[k - r:k + r + 1].max(axis=0).astype(np.float64)
    around = np.median(np.vstack([db[k - 8:k - 3], db[k + 4:k + 9]]).astype(np.float64), axis=0)

    def window_mean(x, first, last, lead=None):        # mean of x[t+first .. t+last], edges repeated
        xp = np.pad(x, 8, mode="edge")
        if lead is not None:
            xp[:8] = lead
        return np.mean([xp[8 + d:8 + d + T] for d in range(first, last + 1)], axis=0)

    after = window_mean(level, 1, 3)
    # the recording is cut to begin at its first sound: before that, this bin was at its quietest
    rise = after - window_mean(level, -4, -2, lead=float(level.min()))
    peak = after - window_mean(around, 1, 3)
    rise[peak < cfg.raw_peak_db] = 0.0
    frames = [f for f in _peaks(rise, cfg.raw_rise_db, cfg.same_pitch_gap) if f + cfg.min_frames_after <= T]
    return frames, after, peak


def fundamental(bundle, pitch: int, f: int) -> tuple[float, float] | None:
    """(level just after frame f, rise from just before) in dB at the key's own fundamental, from the
    raw spectrum. None outside the spectrum or too close to the start."""
    db = getattr(bundle.audio, "stft_db", None)
    if db is None:
        return None
    n_bins, T = db.shape
    hz = 440.0 * 2.0 ** ((pitch - 69) / 12.0)
    k = int(round(hz * (n_bins - 1) * 2 / bundle.audio.sample_rate))
    if k < 2 or k + 2 >= n_bins or f < 7 or f + 4 > T:
        return None
    r = 1 if k * (2.0 ** (1.0 / 12.0) - 1.0) >= 3.0 else 0      # as in _spectral_jumps
    level = db[k - r:k + r + 1].max(axis=0)
    after = float(level[f + 1:f + 4].mean())
    return after, after - float(level[f - 6:f - 2].mean())


def raw_strikes(bundle, pitch: int, cfg: V3Config) -> list[int]:
    """Frames where a strike of `pitch` is read straight off the spectrum: its fundamental jumps by
    raw_rise_db and then stands raw_peak_db above the neighbouring bins.

    The lower note of an octave or fifth: no other score note puts a peak at its fundamental, but
    its attack can read weak beside the louder upper note. The jump counts when the front end
    hears the note sounding afterwards.

    The upper note: its fundamental is also a partial of the lower key and jumps every time that
    key is struck. Where the score also has the lower key without `pitch`, a jump while the lower
    key sounds must show more than that key alone gives: one of the first two partials of `pitch`
    at least raw_rise_db higher, relative to the lower key's fundamental, than at the leanest
    strike of that key in this recording. (A key that is never written alone, as in a scale in
    octaves, leaves nothing to compare with: the jump is taken as it is.)"""
    hz = stft_features.midi_to_frequency
    frames, after, _ = _spectral_jumps(bundle, hz(pitch), cfg)
    if not frames:
        return []
    T = bundle.audio.num_frames

    def heard(p: int, f: int) -> bool:
        pres = bundle.candidate_scores[p][f + 1:min(T, f + 1 + cfg.presence_window)]
        return pres.size > 0 and float(np.median(pres)) >= cfg.match_min_salience

    onsets = [seg for seg in bundle.segments if seg.onset_pitches]
    lower = [p for p in sorted({p for seg in onsets for p in seg.onset_pitches}) if pitch - p in OVERTONE_OFFSETS]
    if not lower:
        return [f for f in frames if heard(pitch, f)]
    partials = [a for a in (after, _spectral_jumps(bundle, 2 * hz(pitch), cfg)[1]) if a is not None]
    for p in lower:
        if all(pitch in seg.expected_pitches for seg in onsets if p in seg.onset_pitches):
            continue
        jumps_p, after_p, peak_p = _spectral_jumps(bundle, hz(p), cfg)
        struck_p = [g for g in jumps_p if heard(p, g)]
        if not struck_p:
            continue
        lean = [min(a[g] - after_p[g] for g in struck_p) for a in partials]

        def explained(f: int) -> bool:   # the lower key is struck with it and accounts for the jump
            t = min(struck_p, key=lambda g: abs(g - f))         # (a key left ringing only decays)
            return abs(t - f) <= cfg.cluster_gap and max(
                a[t] - after_p[t] - l for a, l in zip(partials, lean)) < cfg.raw_rise_db

        frames = [f for f in frames if not explained(f)]
    return frames


def restruck(bundle, pitch: int, f: int, cfg: V3Config) -> bool:
    """Was `pitch`, already ringing, struck again at frame f?"""
    pres, T = bundle.candidate_scores[pitch], bundle.audio.num_frames
    db, en = getattr(bundle.audio, "stft_db", None), getattr(bundle, "frame_energy", None)
    if db is None or en is None or f < 6 or f + 4 > T:
        return False
    if min(float(np.median(pres[f - 6:f - 1])), float(np.median(pres[f + 1:f + 1 + cfg.presence_window]))) < cfg.restrike_presence:
        return False
    n_bins = db.shape[0]
    k = int(round(stft_features.midi_to_frequency(pitch) * (n_bins - 1) * 2 / bundle.audio.sample_rate))
    if k < 1 or k + 2 > n_bins:
        return False
    level = db[k - 1:k + 2].max(axis=0).astype(np.float64)
    if level[f + 1:f + 4].mean() - level[f - 4:f - 1].mean() < cfg.restrike_rise_db:
        return False
    before = float(en[f - 4:f - 1].min())
    if before > 1.0 - cfg.restrike_energy:             # the energy scale tops out at the recording's loud
        return True                                    # level: there it cannot show the rise, so it is not asked
    return float(en[f + 1:f + 4].max()) - before >= cfg.restrike_energy


def key_scales(events: list[Event], cfg: V3Config) -> dict[int, float]:
    """Per key, the factor that brings its strikes to level_reference in this recording.

    The level of a key is read off the clear strikes found without any correction (the loudest key
    of each event, attack at full strength). A key played once or never borrows from its neighbours
    in the register, then from the whole recording, so one faint strike cannot set its own standard."""
    seen: dict[int, list[float]] = {}
    for ev in events:
        if ev.primary and ev.pitch in ev.full and ev.pitch not in ev.raw_only:
            seen.setdefault(ev.pitch, []).append(ev.primary[0][1])
    if not seen:
        return {}
    overall = float(np.median([x for v in seen.values() for x in v]))

    def around(p: int) -> float:
        for span in (cfg.level_span, 2 * cfg.level_span + 1):
            near = [x for q, v in seen.items() if q != p and abs(q - p) <= span for x in v]
            if near:
                return float(np.median(near))
        return overall

    def scale(p: int) -> float:
        own = seen.get(p, [])
        level = (len(own) * float(np.median(own)) if own else 0.0) + cfg.level_prior * around(p)
        level /= len(own) + cfg.level_prior
        return float(np.clip(cfg.level_reference / max(level, 1e-6), cfg.level_scale_min, cfg.level_scale_max))

    pitches = set(seen) | {q for ev in events for q in ev.strikes}
    return {p: scale(p) for p in pitches}


def detect_events(bundle, cfg: V3Config, scales: dict[int, float] | None = None) -> list[Event]:
    """Strike events. With `scales` (see key_scales) every salience is relative to its key's level."""
    T = bundle.audio.num_frames
    full: set[tuple[int, int]] = set()                 # (frame, pitch) whose attack counted at full strength
    cands: list[tuple[int, int, float]] = []          # (frame, pitch, salience)
    raw: set[tuple[int, int]] = set()
    energy = getattr(bundle, "frame_energy", None) if cfg.strike_needs_sound else None

    def silent(f: int) -> bool:
        # presence is a share of what sounds, so it reads high in silence too. Frame energy runs from
        # 0 (the recording's quiet floor) to 1 (its loud passages): nothing is struck at the floor
        return energy is not None and float(np.max(energy[f:f + 4])) < cfg.strike_energy

    def partial_restrike(p: int, q: int, f: int) -> bool:
        # key p sounds before and after frame f; its partial q has no plain attack of its own and is
        # no stronger after f than before
        pres_p, pres_q = bundle.candidate_scores.get(p), bundle.candidate_scores[q]
        if pres_p is None or f < 6 or float(bundle.candidate_onset_scores[q][f]) >= cfg.onset_full:
            return False
        before = [float(np.median(x[f - 6:f - 1])) for x in (pres_p, pres_q)]
        after = [float(np.median(x[f + 1:min(T, f + 1 + cfg.presence_window)])) for x in (pres_p, pres_q)]
        return min(before[0], after[0]) >= cfg.restrike_presence and after[1] <= cfg.overtone_ratio * before[1]

    for q in harmonic_score_pitches(bundle):
        for f in raw_strikes(bundle, q, cfg):
            if not silent(f):
                cands.append((f, q, cfg.match_min_salience))
                raw.add((f, q))
    for p, onset in bundle.candidate_onset_scores.items():
        pres = bundle.candidate_scores[p]
        for f in _peaks(onset, min(cfg.onset_floor, cfg.restrike_floor), cfg.same_pitch_gap):
            if f + cfg.min_frames_after > T or silent(f):
                continue
            gain = float(np.clip((float(onset[f]) - cfg.onset_floor) / (cfg.onset_full - cfg.onset_floor), 0.0, 1.0))
            if gain < 1.0 and restruck(bundle, p, f, cfg):
                gain = 1.0
            if gain <= 0.0:
                continue
            s = gain * float(np.median(pres[f + 1:min(T, f + 1 + cfg.presence_window)]))
            if s >= cfg.min_salience:
                cands.append((f, p, s))
                if gain >= 1.0:
                    full.add((f, p))
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
        # overtones, ghosts and leakage were sorted out on the measured levels; from here on a strike
        # is worth its level relative to that key
        if scales:
            seen_raw = {q for q in strikes if q not in confirmed}       # those carry a nominal level already
            strikes = {q: s * (1.0 if q in seen_raw else scales.get(q, 1.0)) for q, s in strikes.items()}
        # a key struck again while it still rings may show the attack only at its upper partials (its
        # fundamental is loud already): a jump at a partial of a sounding key that leaves the partial
        # no stronger than it was before is that key struck again, not a new note
        for q in sorted(kept):
            src = next((q - off for off in OVERTONE_OFFSETS
                        if q in confirmed and q - off not in strikes and partial_restrike(q - off, q, frames[q])), None)
            if src is not None:
                kept[kept.index(q)] = src
                explained[q], strikes[src], frames[src] = src, strikes[q], frames[q]
                confirmed.add(src)
                if (frames[q], q) in full:
                    full.add((frames[q], src))
        fund = {q: fundamental(bundle, q, frames[q]) for q in strikes}
        rise = {q: v[1] for q, v in fund.items() if v is not None}
        phantom: dict[int, int] = {}
        for q in kept:
            if fund.get(q) is None:
                continue
            partner = next((k for k in sorted(kept, key=lambda k: -(fund[k][0] if fund.get(k) else -1e9))
                            if k != q and k not in phantom and abs(q - k) in PHANTOM_OFFSETS and fund.get(k)
                            and fund[k][0] - fund[q][0] >= cfg.phantom_gap_db), None)
            if partner is not None:
                phantom[q] = partner
        # strongest first, phantoms last: the event's own key is the strongest key that really sounded
        primary = sorted(((q, strikes[q]) for q in kept), key=lambda x: (x[0] in phantom, -x[1]))
        events.append(Event(frames[primary[0][0]], strikes, primary, explained, set(strikes) - confirmed,
                            {q for q in strikes if (frames[q], q) in full}, phantom, rise))
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
    """Cost of resuming at each earlier position j (indexed by j). Learners restart where the music
    restarts: at a bar line far more often than mid-bar. Which jumps are allowed at all (at least
    back_min_notes replayed, within back_bars bars or to the beginning) is decided in align()."""
    return np.where([p.bar_start for p in positions], cfg.back_cost_bar, cfg.back_cost_any).astype(float)


def back_window(positions: list[Position], cfg: V3Config) -> np.ndarray:
    """[j', j] True where a restart from j' to the earlier position j is allowed. Dense: for tests."""
    J = len(positions)
    jj = np.arange(J)
    bar = np.cumsum([p.bar_start for p in positions])
    k = jj[:, None] - jj[None, :]                                       # notes gone back, j' - j
    near = (bar[:, None] - bar[None, :] <= cfg.back_bars) | (jj[None, :] == 0)
    return (k >= max(1, cfg.back_min_notes - 1)) & near


# ---------------------------------------------------------------- emission
def struck(event: Event, pitch: int, expected: tuple[int, ...], cfg: V3Config) -> float:
    """Salience with which `pitch` was struck in `event`, 0 if it was not. A pitch removed as an
    overtone still counts when the note that explains it is itself expected here (octaves, fifths)."""
    s = event.strikes.get(pitch, 0.0)
    if s < cfg.match_min_salience:
        return 0.0
    if pitch in event.phantom and event.phantom[pitch] not in expected:
        return 0.0                         # borrowed from a louder key the score does not want here
    src, hops = event.explained_by.get(pitch), 0
    while src is not None and src in event.explained_by and hops < 8:
        src, hops = event.explained_by[src], hops + 1
    if src is None or src in expected:
        return s
    # the explaining key was only ringing (held from before) while this one was freshly struck
    if event.rise.get(src, 99.0) < cfg.ring_rise_db <= event.rise.get(pitch, -99.0):
        return s
    return 0.0


def emission_cost(event: Event, pos: Position, cfg: V3Config) -> float:
    missing = [p for p in pos.pitches if struck(event, p, pos.pitches, cfg) <= 0.0]
    if not missing:
        return 0.0
    if len(missing) < len(pos.pitches):
        return cfg.chord_missing_cost * len(missing)
    if event.salience < cfg.match_min_salience:
        return np.inf
    weak = max(0.0, cfg.sub_min_salience - event.salience) / max(cfg.sub_min_salience - cfg.match_min_salience, 1e-6)
    near = min(abs(event.pitch - p) for p in pos.pitches) <= 2
    return ((cfg.sub_cost_near if near else cfg.sub_cost) + cfg.sub_weak_penalty * weak
            + cfg.chord_missing_cost * (len(missing) - 1))


def extra_cost(event: Event, cfg: V3Config) -> float:
    w = np.clip((event.salience - cfg.extra_weak_salience)
                / (cfg.extra_strong_salience - cfg.extra_weak_salience), 0.0, 1.0)
    return float(cfg.extra_cost_weak + w * (cfg.extra_cost - cfg.extra_cost_weak))


# ---------------------------------------------------------------- alignment (Viterbi over events)
def _timing(ioi_s: float, beats: np.ndarray, spb: float, cfg: V3Config) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(max(ioi_s, 1e-3) / (beats * spb))
    return np.minimum(cfg.timing_weight * r * r, cfg.timing_cap)


def _emissions(events: list[Event], positions: list[Position], cfg: V3Config) -> np.ndarray:
    """[e, j] emission cost. It depends only on the pitches a position expects, so it is computed
    once per distinct pitch set (a long piece reuses a few dozen of them)."""
    cols: dict[tuple, np.ndarray] = {}
    for pos in positions:
        if pos.pitches not in cols:
            cols[pos.pitches] = np.array([emission_cost(ev, pos, cfg) for ev in events])
    return np.stack([cols[pos.pitches] for pos in positions], axis=1)


def _backtrack(bp: np.ndarray, total: np.ndarray) -> tuple[list[Match], float]:
    e, j = np.unravel_index(int(np.argmin(total)), total.shape)
    cost = float(total[e, j])
    path = []
    while e >= 0:
        e0, j0 = bp[e, j]
        kind = "next" if e0 < 0 or j == j0 + 1 else ("back" if j <= j0 else "skip")
        path.append(Match(int(e), int(j), kind))
        e, j = int(e0), int(j0)
    return path[::-1], cost


def align(events: list[Event], positions: list[Position], cfg: V3Config,
          spb: np.ndarray, spf: float, back_cost=None) -> tuple[list[Match], float]:
    """Cheapest explanation of the events by the score. dp[e, j] = best cost of a path whose last
    matched pair is (event e, position j).

    Only the moves a player makes are searched: the next note, a forward jump over at most
    max_skip notes, and (with back_cost, from back_costs()) a restart at least back_min_notes back
    and at most back_bars bars back, or to the very beginning. Each event therefore looks at a
    fixed number of earlier positions, and the work grows with the length of the piece. For pieces
    shorter than the window this is exactly align_dense()."""
    E, J = len(events), len(positions)
    if E == 0 or J == 0:
        return [], float(min(J * cfg.skip_cost, cfg.stop_cost) if J else 0.0)
    beat = np.array([p.beat for p in positions], dtype=float)
    t = np.array([ev.frame for ev in events]) * spf
    emit = _emissions(events, positions, cfg)
    xc = np.array([extra_cost(ev, cfg) for ev in events])
    xcum = np.concatenate([[0.0], np.cumsum(xc)])
    jj = np.arange(J)
    own_note = np.diff(beat, append=beat[-1] + 1.0)                     # beats of each note itself
    kmin = max(1, cfg.back_min_notes - 1)
    back = back_cost is not None and J > 1
    bar = np.cumsum([p.bar_start for p in positions])

    # columns are moves "from j' = j - d", ordered by increasing j' (ties go to the earliest j',
    # as in align_dense): forward jumps d = F..1, then restarts d = -kmin..-K
    F = min(cfg.max_skip + 1, J - 1)
    K = 0
    if back:
        reach = np.searchsorted(bar, bar + cfg.back_bars, side="right") - 1 - jj   # last j' per target
        K = int(max(0, reach.max()))
    offs = np.concatenate([np.arange(F, 0, -1), -np.arange(kmin, K + 1)]).astype(int)
    S = jj[:, None] - offs[None, :]                                      # [j, c] source position j'
    valid = (S >= 0) & (S < J)
    Sc = np.clip(S, 0, J - 1)
    nf = F                                                               # forward columns come first
    skips = offs[:nf] - 1
    skip_c = np.minimum(skips * cfg.skip_cost, cfg.jump_cost + skips * cfg.jump_step)[None, :]
    skipped = (skips >= 1)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_dbeat = np.log(beat[:, None] - beat[Sc[:, :nf]])            # written beats between j' and j
        log_own = np.log(own_note[Sc[:, :nf]])
    tr_back = np.full((J, offs.size - nf), np.inf)
    if back:
        ok = valid[:, nf:] & (bar[Sc[:, nf:]] - bar[:, None] <= cfg.back_bars)
        tr_back[ok] = np.broadcast_to(back_cost[:, None], ok.shape)[ok]

    def timing(ioi_s: float, log_beats: np.ndarray, spb_e: float) -> np.ndarray:
        r = np.log(max(ioi_s, 1e-3)) - np.log(spb_e) - log_beats
        return np.minimum(cfg.timing_weight * r * r, cfg.timing_cap)

    dp = np.full((E, J), np.inf)
    bp = np.full((E, J, 2), -1, dtype=int)
    start = np.minimum(jj * cfg.skip_cost, cfg.jump_cost + jj * cfg.jump_step)
    for e in range(E):
        dp[e] = xcum[e] + start + emit[e]
        for e0 in range(max(0, e - 1 - cfg.max_extras), e):
            between = xcum[e] - xcum[e0 + 1]
            ioi = t[e] - t[e0]
            tr_f = skip_c + timing(ioi, log_dbeat, spb[e])                # time of the skipped notes kept
            tr_f = np.where(skipped, np.minimum(tr_f, skip_c + cfg.skip_time_slack
                                                + timing(ioi, log_own, spb[e])), tr_f)   # ... or dropped
            if e0 < e - 1:                           # extras in between: they may have added their own time
                tr_f = np.minimum(tr_f, skip_c + cfg.extra_time_slack + timing(t[e] - t[e - 1], log_dbeat, spb[e]))
            cand = dp[e0][Sc] + np.concatenate([tr_f, tr_back], axis=1) + between
            cand[~valid] = np.inf
            c = np.argmin(cand, axis=1)
            best = cand[jj, c] + emit[e]
            j0 = S[jj, c]
            if back:                                 # back to the very beginning, from anywhere
                c0 = dp[e0][kmin:] + back_cost[0] + between
                i0 = int(np.argmin(c0))
                best[0], j0[0] = c0[i0] + emit[e][0], kmin + i0
            better = best < dp[e]
            dp[e][better] = best[better]
            bp[e][better, 0] = e0
            bp[e][better, 1] = j0[better]
    left = J - 1 - jj
    end = np.minimum(left * cfg.skip_cost, cfg.stop_cost + left * cfg.jump_step)
    total = dp + (xcum[E] - xcum[1:E + 1])[:, None] + end[None, :]
    if not np.isfinite(total.min()):
        return [], float(xcum[E] + min(J * cfg.skip_cost, cfg.stop_cost))
    return _backtrack(bp, total)


def align_dense(events: list[Event], positions: list[Position], cfg: V3Config,
                spb: np.ndarray, spf: float, back_cost=None) -> tuple[list[Match], float]:
    """Reference: every position may follow every position (work grows with the square of the piece).
    back_cost here is a dense [j', j] matrix (inf where a restart is not allowed). Kept for tests."""
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
    if cfg.relative_levels:
        events = detect_events(bundle, cfg, key_scales(events, cfg))
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
    flags: set = field(default_factory=set)      # late, held_long (see timing_flags, held_flags)
    late_reason: str = ""          # "held" (previous key kept down) | "paused" (silence before the note)
    late_beats: float = 0.0        # how many beats late


@dataclass
class Report:
    verdicts: dict[int, list[PitchVerdict]]            # position index -> one verdict per expected pitch
    extras: list[tuple[int | None, int, int]]          # (position index it follows or None, pitch, frame)
    restarts: list[tuple[int, int]]                    # (position before the jump, position resumed at)
    visits: dict[int, list[int]]                       # position index -> events matched to it, in order
    by_event: dict[int, list[PitchVerdict]] = field(default_factory=dict)   # matched event -> its verdicts
    attempt_of: dict[int, int] = field(default_factory=dict)               # matched event -> attempt (1, 2, ...)
    # mistakes in an attempt the player abandoned by going back: (attempt, position index, verdict).
    # `verdicts` judge each note by the last attempt that reached it.
    earlier: list[tuple[int, int, PitchVerdict]] = field(default_factory=list)


def verdicts_for(event: Event, pos: Position, prev_pitches: tuple[int, ...], cfg: V3Config) -> tuple[list[PitchVerdict], list[int]]:
    """Per-pitch verdicts for one matched (event, position), plus leftover struck pitches."""
    out, used = [], set()
    free = [(q, s) for q, s in event.primary if q not in pos.pitches and q not in event.phantom]
    for p in pos.pitches:
        s = struck(event, p, pos.pitches, cfg)
        if s > 0.0:
            # two keys at once with the stray one at least as strong: the note was not played cleanly
            stray = [(q, sq) for q, sq in free if q not in used and q not in prev_pitches and sq >= cfg.smudge_ratio * s
                     and abs(q - p) <= cfg.hand_span]    # further away it is the other hand's key: an extra
            # an expected pitch seen only as a jump in the spectrum loses to a clearly struck other key
            doubtful = p in event.raw_only and any(sq >= cfg.sub_min_salience for _, sq in stray)
            stray = [x for x in stray if x[1] >= cfg.sub_min_salience] if doubtful else stray
            if (len(pos.pitches) == 1 or doubtful) and stray:
                used.add(stray[0][0])
                out.append(PitchVerdict(p, "wrong_pitch", stray[0][0], event.frame))
            else:
                out.append(PitchVerdict(p, "correct", p, event.frame))
            continue
        floor = cfg.match_min_salience if len(pos.pitches) == 1 else cfg.sub_min_salience
        cand = [(abs(q - p), q) for q, sq in free if q not in used and sq >= floor]
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
    en = getattr(bundle, "frame_energy", None)         # presence is relative: it reads high in silence too
    if en is not None and float(np.median(en[lo:hi])) < cfg.silence_energy:
        return False
    return float(np.mean(pres[lo:hi] >= cfg.sounding_level * level)) >= 0.5


def timing_flags(an: Analysis, rep: "Report", bundle, cfg: V3Config) -> None:
    """Late = the note started late, whatever the reason. The time since the previous matched note
    is compared with the written distance in beats at the local tempo (missed notes in between keep
    their time). A note that follows an extra or stray strike is not judged: that strike took an
    unknown share of the time and is already reported. The reason is kept beside the flag: the
    previous key was held down, or there was a pause."""
    spf = an.seconds_per_frame
    extra_frames = {f for _, _, f in rep.extras}
    for a, c in zip(an.matches, an.matches[1:]):
        if c.kind == "back" or c.position <= a.position:
            continue                                   # a restart explains its own time
        va, vc = rep.by_event[a.event], rep.by_event[c.event]          # this visit, not the note's last one
        if any(v.corrected for v in vc):
            continue                                   # so does a wrong key pressed first
        if any(an.events[k].salience >= cfg.extra_report_salience or an.events[k].frame in extra_frames
               for k in range(a.event + 1, c.event)):
            continue
        if c.position != a.position + 1 and not cfg.late_after_missed:
            continue
        fa, fc = an.events[a.event].frame, an.events[c.event].frame
        beats = an.positions[c.position].beat - an.positions[a.position].beat
        delay = (fc - fa) * spf / an.seconds_per_beat[c.event] - beats
        if delay < max(cfg.late_beats, cfg.late_fraction * beats):
            continue
        held = any(v.played is not None and still_sounding(bundle, v.played, fa, fc, cfg) for v in va)
        for v in vc:
            v.flags.add("late")
            v.late_reason = "held" if held else "paused"
            v.late_beats = float(delay)


def held_evidence(an: Analysis, rep: "Report", bundle, cfg: V3Config) -> list[tuple[PitchVerdict, float, bool, bool]]:
    """For every pair of consecutive notes whose previous key can be judged:
    (verdict of the previous note, how far in dB its key's fundamental fell from just before the
    next strike to the lowest it reached within held_after_s after it (a held key only fades; one
    whose sound comes back was struck or rang again), is the key still sounding after the wait,
    did the next note come on time).

    Not judged: the same key twice, chords, a note the score keeps sounding, a wrong or missed note
    on either side, a restart or a clear stray strike in between, a key already gone before the
    next strike, a next note shorter than the wait, and a key whose fundamental the notes that
    follow also fill (an overtone of theirs, or too close in frequency to read apart)."""
    db = getattr(bundle.audio, "stft_db", None)
    if db is None or not rep.by_event:
        return []
    T, n_bins = bundle.audio.num_frames, db.shape[0]
    wait = max(1, int(round(cfg.held_after_s / an.seconds_per_frame)))
    extra_frames = {f for _, _, f in rep.extras}

    def key_bin(pitch: int) -> int:
        return int(round(stft_features.midi_to_frequency(pitch) * (n_bins - 1) * 2 / bundle.audio.sample_rate))

    out = []
    for a, c in zip(an.matches, an.matches[1:]):
        if c.kind == "back" or c.position != a.position + 1:
            continue
        va, vc = rep.by_event[a.event], rep.by_event[c.event]
        if len(va) != 1 or len(vc) != 1 or va[0].label != "correct" or vc[0].label != "correct":
            continue
        p = va[0].pitch
        written = set(bundle.segments[an.positions[c.position].segment_index].expected_pitches) | {vc[0].pitch}
        k = key_bin(p)
        if p in written or k < 1 or k + 2 > n_bins:
            continue
        # an overtone of a following note lands on this key's fundamental; the main lobe of a
        # neighbour closer than 3 bins reaches the bins read here
        if any(p - q in OVERTONE_OFFSETS or abs(key_bin(q) - k) < 3 for q in written):
            continue
        if any(an.events[e].salience >= cfg.extra_report_salience or an.events[e].frame in extra_frames
               for e in range(a.event + 1, c.event)):
            continue
        fa, fc = an.events[a.event].frame, an.events[c.event].frame
        following = an.events[c.event + 1].frame if c.event + 1 < len(an.events) else T
        if fc < 3 or fc + wait + 2 > min(following, T) or not still_sounding(bundle, p, fa, fc, cfg):
            continue
        level = db[k - 1:k + 2].max(axis=0).astype(np.float64)
        lowest = float(np.convolve(level[fc:fc + wait + 2], np.ones(3) / 3, mode="valid").min())
        drop = float(level[fc - 3:fc].mean()) - lowest
        out.append((va[0], drop, still_sounding(bundle, p, fa, fc + wait + 3, cfg),
                    not any("late" in v.flags for v in vc)))
    return out


def held_flags(an: Analysis, rep: "Report", bundle, cfg: V3Config) -> None:
    """Held too long = the key was not released when the next note was played on time: two keys
    sound together where the score has one after the other. The held note gets the flag. (A key
    kept down while the next note comes late is `late` on that next note: see timing_flags.)

    A released key is damped at once, a held one goes on fading slowly. How far a key falls in
    held_after_s depends on the piano, the room and the phone, so each pair is compared with the
    recording itself: the usual fall is the median over its judged pairs, nearly all of them
    released. A note is held when its key is still sounding after the wait and has fallen by less
    than held_drop_share of the usual. A recording where most keys keep sounding (pedal, strong
    room ring, everything overlapped) has no released notes to compare with: nothing is flagged."""
    pairs = held_evidence(an, rep, bundle, cfg)
    if not pairs or 2 * sum(sounding for _, _, sounding, _ in pairs) > len(pairs):
        return
    usual = float(np.median([drop for _, drop, _, _ in pairs]))
    for v, drop, sounding, on_time in pairs:
        if sounding and on_time and drop < cfg.held_drop_share * usual:
            v.flags.add("held_long")


def report(an: Analysis, cfg: V3Config | None = None, bundle=None) -> Report:
    """Turn the alignment into verdicts.

    Attempts: the take starts in attempt 1 and every restart starts the next one. Each note is
    judged by the last attempt that reached it (played it, or skipped over it); what went wrong in
    earlier attempts is kept in `earlier`, so a player who went back is told what made them stop.

    An unmatched event is one of three things: a re-strike of the note just played or a double
    strike of the note that follows (an extra note); a wrong key pressed before the right one
    (the following note is reported as wrong pitch, corrected); or noise (too faint to report)."""
    cfg = cfg or V3Config()
    verdicts: dict[int, list[PitchVerdict]] = {}
    visits: dict[int, list[int]] = {}
    extras, restarts = [], []
    matched = {m.event: m for m in an.matches}
    last_pos, prev_pitches, last_ev = None, (), None
    attempt, by_event, attempt_of = 1, {}, {}
    touches: dict[int, list[tuple[int, list[PitchVerdict] | None]]] = {}   # position -> (attempt, verdicts or None = skipped)
    pending: list[Event] = []                          # unmatched events since the last match that may be notes

    def clear(x: Event, neighbours: list[Event]) -> bool:
        """Loud enough to report: plainly, or as a repeat of a neighbouring note's key."""
        if x.salience >= cfg.extra_report_salience:
            return True
        ref = max((dict(n.primary).get(x.pitch, 0.0) for n in neighbours if n is not None), default=0.0)
        return ref > 0.0 and x.salience >= cfg.restrike_report_ratio * ref
    for e, ev in enumerate(an.events):
        m = matched.get(e)
        if m is None:
            if ev.salience >= cfg.match_min_salience:
                pending.append(ev)
            continue
        pos = an.positions[m.position]
        if m.kind == "back" and last_pos is not None:
            restarts.append((last_pos, m.position))
            attempt += 1
        passed = range(last_pos + 1, m.position) if last_pos is not None and m.position > last_pos + 1 else \
            range(0, m.position) if last_pos is None else range(0)
        for q in passed:                                 # notes jumped over in this attempt
            touches.setdefault(q, []).append((attempt, None))
        vs, left = verdicts_for(ev, pos, prev_pitches, cfg)
        prev_set = set(an.positions[last_pos].pitches) if last_pos is not None else set()
        # a second key struck together with the note (a real attack of its own, as clear as any
        # reported extra, and not the previous note still ringing) is an extra note
        # (only beside a correctly struck note: next to a wrong one the leftover key is ambiguous; and
        # a key an overtone interval from an expected one may be that note's partial or its source)
        if all(v.label == "correct" for v in vs):
            extras += [(last_pos, q, ev.frame) for q in left
                       if q in ev.full and q not in prev_pitches and ev.strikes[q] >= cfg.extra_report_salience
                       and not any(abs(q - p) in OVERTONE_OFFSETS for p in pos.pitches)]
        for x in pending:
            if not clear(x, [last_ev, ev]):
                continue
            target = min(vs, key=lambda v: abs(v.pitch - x.pitch))
            lead = (ev.frame - x.frame) * an.seconds_per_frame / an.seconds_per_beat[e]
            if (x.pitch in prev_set or x.pitch in pos.pitches or m.kind == "back" or target.label != "correct"
                    or lead > cfg.correction_max_beats):
                extras.append((last_pos, x.pitch, x.frame))
            else:
                target.label, target.played, target.corrected = "wrong_pitch", x.pitch, True
        pending = []
        by_event[e], attempt_of[e] = vs, attempt
        touches.setdefault(m.position, []).append((attempt, vs))
        visits.setdefault(m.position, []).append(e)
        last_pos, last_ev = m.position, ev
        prev_pitches = tuple(v.played for v in vs if v.played is not None)
    extras += [(last_pos, x.pitch, x.frame) for x in pending if clear(x, [last_ev])]
    earlier: list[tuple[int, int, PitchVerdict]] = []
    for pos in an.positions:
        ts = touches.get(pos.index, [])
        for att, vs in ts[:-1]:                          # every attempt but the last that reached this note
            for v in (vs if vs is not None else [PitchVerdict(p, "missed") for p in pos.pitches]):
                if v.label != "correct":
                    earlier.append((att, pos.index, v))
        last = ts[-1][1] if ts else None
        verdicts[pos.index] = last if last is not None else [PitchVerdict(p, "missed") for p in pos.pitches]
    earlier.sort(key=lambda x: (x[0], x[1]))
    rep = Report(verdicts, extras, restarts, visits, by_event, attempt_of, earlier)
    if bundle is not None:
        timing_flags(an, rep, bundle, cfg)
        held_flags(an, rep, bundle, cfg)
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
            if "late" in v.flags:
                why = {"held": "the previous key was held down", "paused": "after a pause"}.get(v.late_reason, "")
                again += f"  LATE by {v.late_beats:.1f} beats" + (f" ({why})" if why else "")
            if "held_long" in v.flags:
                again += "  HELD TOO LONG (still sounding under the next note)"
            print(f"  note {pos.index + 1:3d}  beat {pos.beat:5.2f}  {midi_to_note_name(v.pitch):<4} {when}  {what}{again}")
    for after, pitch, frame in rep.extras:
        where = f"after note {after + 1}" if after is not None else "before the first note"
        print(f"  EXTRA {midi_to_note_name(pitch)} at {frame * spf:.2f}s ({where})")
    for k, (a, b) in enumerate(rep.restarts, 1):
        why = [f"note {pi + 1} " + (f"WRONG NOTE (played {midi_to_note_name(v.played or 0)})" if v.label == "wrong_pitch"
                                    else "MISSED") for att, pi, v in rep.earlier if att == k]
        print(f"  WENT BACK from note {a + 1} to note {b + 1}"
              + (f", after: {'; '.join(why)}" if why else " (no mistake heard before going back)"))
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
