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

    def config(self) -> dict:
        from stft_hmm_v2 import TimingConfig
        return asdict(TimingConfig())

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


ADAPTERS = {
    "v2": V2Adapter,
}


def get_adapter(name: str):
    if name not in ADAPTERS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(ADAPTERS)}")
    return ADAPTERS[name]()
