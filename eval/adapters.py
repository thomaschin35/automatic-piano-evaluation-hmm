"""
adapters.py
Model registry. Each adapter turns one recording into a TakePrediction
(see eval/contract.py). The "v2" adapter wraps stft_hmm_v2, which grades
time slices, and maps each score note to the slice where it starts.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import contextlib
import io

import matplotlib
matplotlib.use("Agg")

from eval.contract import NotePrediction, TakePrediction
from eval.pieces import PieceSpec


def segments_to_note_predictions(results, segments, spec: PieceSpec,
                                 seconds_per_frame: float, offset_s: float) -> list[NotePrediction]:
    """Map v2 SegmentResults onto score notes via each note's onset segment."""
    by_index = {r.segment_index: r for r in results}
    preds = []
    notes = spec.notes()
    # group notes by the segment they start in, so a two-note chord can share out wrong pitches
    groups: dict[int, list] = {}
    for n in notes:
        seg = next(s for s in segments if abs(s.start_beat - n.start_beat) < 1e-6)
        groups.setdefault(seg.index, []).append(n)
    for seg_idx, group in groups.items():
        r = by_index[seg_idx]
        onset = r.start_frame * seconds_per_frame + offset_s
        substituted = dict(r.substitutes)          # expected pitch -> pitch played instead
        for n in group:
            if n.pitch in r.detected_pitches:
                preds.append(NotePrediction(n.id, "correct", n.pitch, onset))
            elif n.pitch in substituted:
                preds.append(NotePrediction(n.id, "wrong_pitch", substituted[n.pitch], onset))
            else:
                preds.append(NotePrediction(n.id, "missed", None, onset))
    return preds


class V2Adapter:
    name = "v2"
    capabilities = frozenset({"wrong_pitch", "missed"})
    onset_only = {"pitch_weight": 0.0}       # overrides for the "can onsets alone align?" check

    def config(self) -> dict:
        from stft_hmm_v2 import TimingConfig
        return asdict(TimingConfig())

    def analyze(self, bundle, overrides: dict | None = None):
        import dataclasses
        import stft_hmm_v2 as m
        cfg = dataclasses.replace(m.TimingConfig(), **(overrides or {}))
        alignment, _ = m.create_recording_alignment(bundle, cfg)
        states, A = m.build_graph(alignment, cfg)
        path = m.viterbi(m.log_emissions(states, bundle, cfg, alignment), A)
        iv = m.path_to_intervals(path, states, len(alignment), bundle.audio.num_frames)
        return m.grade(iv, alignment, bundle, cfg), alignment

    def predict_bundle(self, bundle, piece: PieceSpec, overrides: dict | None = None) -> TakePrediction:
        """Prediction from precomputed features (bundle.score must already be `piece`)."""
        results, _ = self.analyze(bundle, overrides)
        notes = segments_to_note_predictions(
            results, bundle.segments, piece,
            bundle.audio.seconds_per_frame, bundle.audio.trim_offset_seconds)
        return TakePrediction(notes=notes, capabilities=self.capabilities)

    def predict(self, audio_path: str, piece: PieceSpec, out_dir: Path) -> TakePrediction:
        import stft_hmm_v2
        out_dir.mkdir(parents=True, exist_ok=True)
        plot = out_dir / "alignment.png"
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            bundle, results = stft_hmm_v2.run_pipeline(str(audio_path), piece.name, str(plot))
        (out_dir / "model_stdout.txt").write_text(log.getvalue())
        notes = segments_to_note_predictions(
            results, bundle.segments, piece,
            bundle.audio.seconds_per_frame, bundle.audio.trim_offset_seconds)
        return TakePrediction(notes=notes, capabilities=self.capabilities,
                              artifacts={"plot": str(plot), "stdout": str(out_dir / "model_stdout.txt")})


class V3Adapter:
    name = "v3"
    capabilities = frozenset({"wrong_pitch", "missed", "extra", "late"})      # held_long (overlap): not yet
    onset_only = {"sub_cost": 0.0}           # pitch-blind alignment: any event may stand for any note

    def config(self) -> dict:
        from stft_hmm_v3 import V3Config
        return asdict(V3Config())

    def predict_bundle(self, bundle, piece: PieceSpec, overrides: dict | None = None) -> TakePrediction:
        import dataclasses
        import stft_hmm_v3 as m
        cfg = dataclasses.replace(m.V3Config(), **(overrides or {}))
        an = m.analyze(bundle, cfg, piece.beats_per_measure, piece.pickup_beats)
        rep = m.report(an, cfg, bundle)
        return report_to_prediction(an, rep, piece, bundle.audio.trim_offset_seconds, self.capabilities)

    def predict(self, audio_path: str, piece: PieceSpec, out_dir: Path) -> TakePrediction:
        import stft_hmm_v3 as m
        from eval.cache import load_bundle
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle = load_bundle(audio_path, piece.name)
        cfg = m.V3Config()
        an = m.analyze(bundle, cfg, piece.beats_per_measure, piece.pickup_beats)
        rep = m.report(an, cfg, bundle)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            m.print_report(an, rep)
        (out_dir / "model_stdout.txt").write_text(log.getvalue())
        pred = report_to_prediction(an, rep, piece, bundle.audio.trim_offset_seconds, self.capabilities)
        pred.artifacts = {"stdout": str(out_dir / "model_stdout.txt")}
        self._last = (out_dir, bundle, an, rep, piece)
        return pred

    def plot_take(self, out_dir: Path, rows: list[dict] | None = None) -> Path | None:
        """alignment.png for the take just predicted into out_dir; rows add the gold labels."""
        if getattr(self, "_last", None) is None or self._last[0] != out_dir:
            return None
        import matplotlib.pyplot as plt
        from eval.plots import plot_take
        _, bundle, an, rep, piece = self._last
        fig = plot_take(bundle, an, rep, piece, f"{out_dir.name}  ({piece.name})", rows)
        path = out_dir / "alignment.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path


def report_to_prediction(an, rep, spec: PieceSpec, offset_s: float, capabilities) -> TakePrediction:
    """Map v3 per-position verdicts onto score note ids."""
    from eval.contract import ExtraNotePrediction
    spf = an.seconds_per_frame
    by_beat = {round(p.beat, 4): p for p in an.positions}
    notes_at: dict[int, list] = {}
    preds = []
    for n in spec.notes():
        pos = by_beat[round(n.start_beat, 4)]
        notes_at.setdefault(pos.index, []).append(n)
        v = next(v for v in rep.verdicts[pos.index] if v.pitch == n.pitch)
        onset = v.frame * spf + offset_s if v.frame is not None else None
        flags = frozenset(getattr(v, "flags", ()) or ())
        preds.append(NotePrediction(n.id, v.label, v.played, onset, flags))

    def anchor(pos_index):
        if pos_index is None:
            return None
        ns = notes_at[pos_index]
        return ns[0].id

    extras = [ExtraNotePrediction(anchor(after), pitch, frame * spf + offset_s) for after, pitch, frame in rep.extras]
    restarts = [(notes_at[a][0].id, notes_at[b][0].id) for a, b in rep.restarts]
    from eval.contract import EarlierMistake
    earlier = [EarlierMistake(att, next(n.id for n in notes_at[pi] if n.pitch == v.pitch), v.label,
                              v.played if v.label == "wrong_pitch" else None)
               for att, pi, v in getattr(rep, "earlier", [])]
    return TakePrediction(notes=preds, extras=extras, capabilities=capabilities, restarts=restarts,
                          earlier=earlier)


ADAPTERS = {
    "v2": V2Adapter,
    "v3": V3Adapter,
}


def get_adapter(name: str):
    if name not in ADAPTERS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(ADAPTERS)}")
    return ADAPTERS[name]()
