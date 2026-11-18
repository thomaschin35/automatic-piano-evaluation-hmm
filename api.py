"""
api.py -- FastAPI service wrapping stft_hmm_v2 for Piano Project integration.
Implements schema_version "1.0" from BACKEND_API_INTEGRATION.md.
"""
from __future__ import annotations
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import stft_features
import stft_hmm_v2
from eval.adapters import extras_from_insertions, segments_to_note_predictions
from eval.pieces import load_piece

app = FastAPI(title="Automatic Piano Evaluation API", version="1.0.0")

API_TOKEN = os.getenv("PIANO_EVAL_API_TOKEN", "")
MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB
ALLOWED_EXTS = {".wav", ".mp3", ".m4a"}

# Canonical score registry matching eval/pieces/*.toml and score_models.py
SCORES_META = {
    "ode_melody": {
        "score_id": "ode_melody",
        "version": "1.0",
        "title": "Ode to Joy (Melody)",
        "composer": "Ludwig van Beethoven",
        "section": "Main Theme — bars 1–4",
        "reference_bpm": 120,
        "time_signature": "4/4",
        "difficulty": "beginner",
        "supported_hands": ["rh"],
        "active": True,
    },
    "ode_harmony": {
        "score_id": "ode_harmony",
        "version": "1.0",
        "title": "Ode to Joy (Two Hands)",
        "composer": "Ludwig van Beethoven",
        "section": "Main Theme — bars 1–4",
        "reference_bpm": 120,
        "time_signature": "4/4",
        "difficulty": "beginner",
        "supported_hands": ["both"],
        "active": True,
    },
    "c_major_rh": {
        "score_id": "c_major_rh",
        "version": "1.0",
        "title": "C Major Scale (Right Hand)",
        "composer": "Traditional",
        "section": "1 Octave Ascending",
        "reference_bpm": 120,
        "time_signature": "4/4",
        "difficulty": "beginner",
        "supported_hands": ["rh"],
        "active": True,
    },
    "c_major_both": {
        "score_id": "c_major_both",
        "version": "1.0",
        "title": "C Major Scale (Both Hands)",
        "composer": "Traditional",
        "section": "1 Octave Ascending",
        "reference_bpm": 120,
        "time_signature": "4/4",
        "difficulty": "beginner",
        "supported_hands": ["both"],
        "active": True,
    },
    "e_major_rh": {
        "score_id": "e_major_rh",
        "version": "1.0",
        "title": "E Major Scale (Right Hand)",
        "composer": "Traditional",
        "section": "1 Octave Ascending",
        "reference_bpm": 120,
        "time_signature": "4/4",
        "difficulty": "beginner",
        "supported_hands": ["rh"],
        "active": True,
    },
    "e_major_rh_down": {
        "score_id": "e_major_rh_down",
        "version": "1.0",
        "title": "E Major Scale (Descending)",
        "composer": "Traditional",
        "section": "1 Octave Descending",
        "reference_bpm": 120,
        "time_signature": "4/4",
        "difficulty": "beginner",
        "supported_hands": ["rh"],
        "active": True,
    },
    "canon_lh": {
        "score_id": "canon_lh",
        "version": "1.0",
        "title": "Canon in D (Left Hand)",
        "composer": "Johann Pachelbel",
        "section": "Ground Bass — bars 1–8",
        "reference_bpm": 100,
        "time_signature": "4/4",
        "difficulty": "intermediate",
        "supported_hands": ["lh"],
        "active": True,
    },
    "canon_both": {
        "score_id": "canon_both",
        "version": "1.0",
        "title": "Canon in D (Both Hands)",
        "composer": "Johann Pachelbel",
        "section": "Bars 1–8",
        "reference_bpm": 100,
        "time_signature": "4/4",
        "difficulty": "intermediate",
        "supported_hands": ["both"],
        "active": True,
    },
}

# In-memory store for async polling compatibility
RESULTS_STORE: dict[str, dict] = {}


def _error(status_code: int, code: str, message: str, retryable: bool = False):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
                "request_id": f"req_{uuid.uuid4().hex[:10]}",
            }
        },
    )


def verify_token(authorization: str | None = Header(default=None)):
    if not API_TOKEN:
        return  # Local dev mode if no token is configured
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="UNAUTHORIZED")
    if authorization.removeprefix("Bearer ").strip() != API_TOKEN:
        raise HTTPException(status_code=401, detail="UNAUTHORIZED")


def evaluate_audio_file(audio_path: str, score_id: str, source_name: str, eval_id: str) -> dict:
    piece = load_piece(score_id)
    cfg = stft_hmm_v2.TimingConfig()
    bundle = stft_features.run_stage1_and_stage2(audio_path, score_id)
    results, insertions, tempo_info, _ = stft_hmm_v2.align_bundle(bundle, cfg)

    spf = bundle.audio.seconds_per_frame
    offset = bundle.audio.trim_offset_seconds
    fpb = 60.0 / max(tempo_info["bpm"] * spf, 1e-6)

    note_preds = segments_to_note_predictions(
        results, bundle.segments, piece, spf, offset,
        candidate_scores=bundle.candidate_scores, interior_fpb=fpb, cfg=cfg,
    )
    extras = extras_from_insertions(
        insertions, results, bundle.segments, piece, spf, offset,
        bundle.candidate_onset_scores,
    )

    pred_by_id = {p.note_id: p for p in note_preds}
    seg_by_beat = {round(s.start_beat, 4): s for s in bundle.segments}
    res_by_idx = {r.segment_index: r for r in results}

    events = []
    n_correct = n_wrong = n_missing = 0
    for idx, n in enumerate(piece.notes()):
        p = pred_by_id[n.id]
        seg = seg_by_beat.get(round(n.start_beat, 4))
        r = res_by_idx.get(seg.index) if seg else None

        pred_label = getattr(p, "label", getattr(p, "status", "missed"))
        played_pitch = getattr(p, "pitch", getattr(p, "played_pitch", None))

        if pred_label == "correct":
            status = "ok"
            n_correct += 1
        elif pred_label == "wrong_pitch":
            status = "wrong"
            n_wrong += 1
        else:
            status = "missing"
            n_missing += 1

        start_s = round(r.start_frame * spf + offset, 3) if (r and r.start_frame >= 0) else None
        end_s = round(r.end_frame * spf + offset, 3) if (r and r.end_frame >= 0) else None
        conf = (
            float(bundle.candidate_scores[n.pitch][r.start_frame : r.end_frame + 1].mean())
            if (r and r.start_frame >= 0 and n.pitch in bundle.candidate_scores)
            else 0.0
        )

        events.append({
            "event": idx,
            "note_id": n.id,
            "hand": n.hand,
            "bar": int(n.start_beat // 4) + 1,
            "start_beat": n.start_beat,
            "beats": round(n.end_beat - n.start_beat, 3),
            "status": status,
            "flags": sorted(p.flags),
            "expected": [n.pitch],
            "observed": [played_pitch] if played_pitch is not None else [],
            "start_seconds": start_s,
            "end_seconds": end_s,
            "confidence": round(conf, 3),
        })

    expected_total = len(events)
    precision = n_correct / max(1, n_correct + n_wrong + len(extras))
    recall = n_correct / max(1, expected_total)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "schema_version": "1.0",
        "algorithm_version": "stft_hmm_v2.2026.11",
        "evaluation_id": eval_id,
        "source": source_name,
        "score_id": score_id,
        "reference_bpm": SCORES_META[score_id]["reference_bpm"],
        "audio": {
            "duration_seconds": round(bundle.audio.num_frames * spf + offset, 3),
            "trimmed_start_seconds": round(offset, 3),
            "trimmed_end_seconds": round(bundle.audio.num_frames * spf + offset, 3),
        },
        "pitch": {
            "score": round(f1 * 100),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "correct": n_correct,
            "expected": expected_total,
            "missing": n_missing,
            "wrong": n_wrong,
            "insertions": len(extras),
        },
        "timing": {
            "average_bpm": round(tempo_info["bpm"], 1),
            "rhythm_score": round(max(0.0, 1.0 - (n_missing / max(1, expected_total))), 3),
        },
        "events": events,
    }


@app.get("/health")
def health():
    return {"status": "ok", "algorithm_version": "stft_hmm_v2.2026.11"}


@app.get("/v1/scores", dependencies=[Depends(verify_token)])
def list_scores():
    return {"scores": list(SCORES_META.values())}


@app.post("/v1/evaluations", dependencies=[Depends(verify_token)])
async def create_evaluation(
    file: UploadFile = File(...),
    score_id: str = Form(...),
    client_request_id: str | None = Form(default=None),
):
    if score_id not in SCORES_META:
        return _error(400, "UNKNOWN_SCORE", f"Unknown score_id '{score_id}'.")

    suffix = Path(file.filename or "upload.wav").suffix.lower()
    if suffix not in ALLOWED_EXTS:
        return _error(415, "UNSUPPORTED_AUDIO_FORMAT", f"Unsupported format '{suffix}'. Use WAV, MP3, or M4A.")

    data = await file.read()
    if len(data) > MAX_FILE_BYTES:
        return _error(413, "FILE_TOO_LARGE", "Audio file exceeds 25 MB limit.")

    eval_id = f"eval_{uuid.uuid4().hex[:12]}"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        result = evaluate_audio_file(tmp_path, score_id, file.filename or "upload.wav", eval_id)
        RESULTS_STORE[eval_id] = {
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "result": result,
        }
        # Returns completed status AND inline result so frontend can either read immediately or poll
        return {
            "evaluation_id": eval_id,
            "status": "completed",
            "created_at": RESULTS_STORE[eval_id]["created_at"],
            "result": result,
        }
    except Exception as exc:
        return _error(422, "ALIGNMENT_FAILED", f"Evaluation failed: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)  # Immediate deletion per retention policy


@app.get("/v1/evaluations/{evaluation_id}", dependencies=[Depends(verify_token)])
def get_evaluation_status(evaluation_id: str):
    item = RESULTS_STORE.get(evaluation_id)
    if not item:
        return _error(404, "NOT_FOUND", f"Evaluation '{evaluation_id}' not found.")
    return {"evaluation_id": evaluation_id, "status": item["status"], "progress": 100}


@app.get("/v1/evaluations/{evaluation_id}/result", dependencies=[Depends(verify_token)])
def get_evaluation_result(evaluation_id: str):
    item = RESULTS_STORE.get(evaluation_id)
    if not item:
        return _error(404, "NOT_FOUND", f"Evaluation '{evaluation_id}' not found.")
    return item["result"]