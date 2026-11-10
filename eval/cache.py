"""
cache.py
Feature cache and bundle surgery for fast, repeatable checks.

  load_bundle(audio, piece)   Stage-1/2 features for one recording, pickled under eval/cache/
                              (keyed by the audio file and the stft_features.py source).
  rescore(bundle, score)      same audio features graded against a different score.
  splice(bundle, frames)      same score, time axis re-indexed (delete / repeat / reorder frames).
"""
from __future__ import annotations
from pathlib import Path
import copy
import dataclasses
import hashlib
import pickle

import numpy as np

import stft_features
from score_models import ScoreModel, slice_score_to_segments

from eval import EVAL_DIR, REPO_DIR

CACHE_DIR = EVAL_DIR / "cache"


def _features_version() -> str:
    src = (REPO_DIR / "stft_features.py").read_bytes() + (REPO_DIR / "score_models.py").read_bytes()
    return hashlib.sha1(src).hexdigest()[:10]


def load_bundle(audio_path: str | Path, piece_name: str):
    """Cached stft_features.run_stage1_and_stage2. The waveform and linear STFT are dropped."""
    audio_path = Path(audio_path)
    st = audio_path.stat()
    key = hashlib.sha1(f"{audio_path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{piece_name}".encode()).hexdigest()[:12]
    path = CACHE_DIR / _features_version() / f"{audio_path.stem}_{key}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    bundle = stft_features.run_stage1_and_stage2(str(audio_path), piece_name)
    a = bundle.audio
    bundle.audio = dataclasses.replace(
        a, waveform=np.zeros(0, dtype=np.float32), stft_magnitude=np.zeros((0, 0), dtype=np.float32),
        stft_db=a.stft_db.astype(np.float32), onset_stft_db=np.zeros((0, 0), dtype=np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    return bundle


def rescore(bundle, score: ScoreModel):
    """Copy of `bundle` whose score-dependent fields follow `score` (audio features shared)."""
    b = copy.copy(bundle)
    b.score = score
    b.segments = slice_score_to_segments(score)
    pitches = sorted({p for s in b.segments for p in s.expected_pitches})
    b.note_scores = {p: b.candidate_scores[p] for p in pitches}
    b.note_onset_scores = {p: b.candidate_onset_scores[p] for p in pitches}
    b.unexpected_pitch_scores = stft_features.calculate_unexpected_pitch_feature(
        b.candidate_scores, set(pitches), b.audio.num_frames)
    b.observations, b.tracked_pitches = stft_features.build_observation_matrix(
        b.note_scores, b.note_onset_scores, b.broadband_onset, b.frame_energy, b.unexpected_pitch_scores)
    return b


def splice(bundle, frames):
    """Copy of `bundle` whose time axis is `frames` (an index array into the original frames).
    An index of -1 inserts a silent frame."""
    idx = np.asarray(frames, dtype=int)
    silent = idx < 0
    idx = np.where(silent, 0, idx)

    def take(v, fill=0.0):
        out = v[..., idx].copy() if v.ndim == 1 else v[:, idx].copy()
        out[..., silent] = fill
        return out

    b = copy.copy(bundle)
    b.audio = dataclasses.replace(bundle.audio, stft_db=take(bundle.audio.stft_db, -90.0))
    b.candidate_scores = {p: take(v) for p, v in bundle.candidate_scores.items()}
    b.candidate_onset_scores = {p: take(v) for p, v in bundle.candidate_onset_scores.items()}
    b.note_scores = {p: b.candidate_scores[p] for p in bundle.note_scores}
    b.note_onset_scores = {p: b.candidate_onset_scores[p] for p in bundle.note_onset_scores}
    b.unexpected_pitch_scores = take(bundle.unexpected_pitch_scores)
    b.broadband_onset = take(bundle.broadband_onset)
    b.frame_energy = take(bundle.frame_energy)
    b.observations = bundle.observations[idx].copy()
    b.observations[silent] = 0.0
    return b
