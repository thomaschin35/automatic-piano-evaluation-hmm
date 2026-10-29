"""
eval  --  Evaluation harness for the piano-error model.

Owns the piece specs (eval/pieces/*.toml), the ground-truth labels
(eval/labels/<piece>/<take>.toml) and the manifest. Models are plugged in
through the per-note prediction contract in eval/contract.py.

Usage: python -m eval.cli --help
"""
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_DIR = EVAL_DIR.parent
