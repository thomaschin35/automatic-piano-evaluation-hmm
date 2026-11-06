"""
cli.py  --  python -m eval.cli <command>

  evaluate [--sheet F]         one step: sync new recordings, import the sheet, run, chart
  sync                         register new recordings (piece from file name), convert,
                               and write labeling-sheet rows for them
  status                       per piece: takes ready, waiting for labels, excluded
  convert                      m4a -> wav for every .m4a take in the manifest (or the files given)
  score <piece>                print the numbered score used by label files
  pieces                       list piece specs
  new-take <audio> --piece P   add a manifest entry and write a label template
  sheet-template [--new]       per-note labeling sheet (TSV) to paste into Google Sheets
  import-sheet <csv|tsv>       write label files from the filled-in sheet
  validate                     check manifest, audio, labels and piece specs
  run [--model v2] [--split dev|test|all] [--piece P] [--take T]
  plots <run>                  overview, confusion and note-map charts for a run
  compare <runA> <runB>        metric deltas and per-note flips between two runs
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from eval import REPO_DIR
from eval.labels import (MANIFEST_PATH, SPLITS, append_manifest_entry, label_template,
                         load_labels, load_manifest)
from eval.pieces import all_pieces, load_piece

SHEETS_DIR = REPO_DIR / "eval" / "sheets"

# ---------------------------------------------------------------- audio
def wav_path_for(m4a: Path) -> Path:
    """.../raw/x.m4a -> .../processed/x.wav (or x.wav alongside when there is no raw/ folder)."""
    m4a = m4a.resolve()
    parts = list(m4a.parts)
    if "raw" in parts:
        i = len(parts) - 1 - parts[::-1].index("raw")
        parts[i] = "processed"
        return Path(*parts).with_suffix(".wav")
    return m4a.with_suffix(".wav")


def model_audio(path: Path) -> Path:
    """Manifest audio may be an .m4a (or other phone format); the model gets a cached .wav conversion."""
    return path if path.suffix.lower() == ".wav" else convert_one(path)


def convert_one(src: Path, force: bool = False) -> Path:
    dst = wav_path_for(src)
    if not force and dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH")
    dst.parent.mkdir(parents=True, exist_ok=True)
    # phone ALAC files end in an empty frame that ffmpeg reports as a decoding error (it only drops
    # ~0.09 s of trailing silence), so only fatal errors are shown; a failed conversion still raises
    subprocess.run(["ffmpeg", "-y", "-loglevel", "fatal", "-i", str(src), "-ac", "1", str(dst)], check=True)
    print(f"converted {src.relative_to(REPO_DIR) if src.is_relative_to(REPO_DIR) else src} -> "
          f"{dst.relative_to(REPO_DIR)}")
    return dst


def cmd_convert(args) -> int:
    sources = [Path(p).resolve() for p in args.files] or sorted(
        {t.audio for t in load_manifest() if t.audio.suffix.lower() == ".m4a" and t.audio.exists()})
    if not sources:
        print("No .m4a takes in the manifest")
    for src in sources:
        convert_one(src, args.force)
    return 0


# ---------------------------------------------------------------- scores
def cmd_pieces(args) -> int:
    for name, spec in all_pieces().items():
        status = "PENDING transcription" if spec.pending else f"{len(spec.notes())} notes, {spec.total_beats:g} beats"
        print(f"{name:<16} {status:<28} {spec.title}")
    return 0


def cmd_score(args) -> int:
    spec = load_piece(args.piece)
    print(f"{spec.title}  ({spec.time_signature[0]}/{spec.time_signature[1]}, {spec.bpm:g} bpm)")
    if spec.pending:
        print("PENDING transcription: no notes yet.")
        return 1
    print(f"{'id':<6}{'pitch':<7}{'measure':<9}{'beat':<7}{'length':<7}")
    for n in spec.notes():
        print(f"{n.id:<6}{n.name:<7}{n.measure:<9}{n.beat_in_measure:<7g}{n.end_beat - n.start_beat:<7g}")
    return 0


# ---------------------------------------------------------------- takes
def cmd_new_take(args) -> int:
    spec = load_piece(args.piece)
    audio = Path(args.audio).resolve()
    if not audio.exists():
        sys.exit(f"Audio not found: {audio}")
    take_id = args.id or audio.stem
    if any(t.id == take_id for t in load_manifest()):
        sys.exit(f"Take id '{take_id}' already in {MANIFEST_PATH.name}; pass --id")
    try:
        rel = audio.relative_to(REPO_DIR)
    except ValueError:
        sys.exit(f"Audio must live inside the repo ({REPO_DIR}), e.g. under recordings/eval/")
    append_manifest_entry(take_id, spec.name, rel.as_posix(), args.split)
    label_path = REPO_DIR / "eval" / "labels" / spec.name / f"{take_id}.toml"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    if label_path.exists():
        print(f"Label file already exists, left unchanged: {label_path.relative_to(REPO_DIR)}")
    else:
        label_path.write_text(label_template(spec))
        print(f"Wrote label template: {label_path.relative_to(REPO_DIR)}")
    print(f"Added take '{take_id}' ({spec.name}, {args.split}) to {MANIFEST_PATH.relative_to(REPO_DIR)}")
    return 0


def cmd_sync(args) -> list:
    """Register every new recording in the intake folders; returns the new Takes."""
    from eval.intake import find_new_recordings
    pieces = all_pieces()
    cands = find_new_recordings(load_manifest(), pieces)
    new = [c for c in cands if c.piece]
    for c in cands:
        if c.problem:
            print(f"SKIP  {c.path.relative_to(REPO_DIR)}: {c.problem}")
    if not new:
        print("No new recordings to register." if not cands else "No new recordings registered.")
        return []
    for c in new:
        print(f"{'would add' if args.dry_run else 'NEW  '} {c.take_id:<24} -> {c.piece}")
    if args.dry_run:
        return []
    for c in new:
        append_manifest_entry(c.take_id, c.piece, c.path.relative_to(REPO_DIR).as_posix(), args.split)
        model_audio(c.path)
    added = [t for t in load_manifest() if t.id in {c.take_id for c in new}]
    SHEETS_DIR.mkdir(exist_ok=True)
    out = SHEETS_DIR / f"to_label_{time.strftime('%Y%m%d-%H%M%S')}.tsv"
    out.write_text(sheet_rows(added, pieces))
    print(f"\nAdded {len(added)} takes to {MANIFEST_PATH.relative_to(REPO_DIR)} (split={args.split}).")
    print(f"Labeling rows: {out.relative_to(REPO_DIR)}\n"
          f"  paste them (without the header row) under the existing rows of the shared sheet,\n"
          f"  then for each take change every 'todo' to correct / wrong_pitch / missed.")
    return added


def cmd_status(args) -> int:
    pieces = all_pieces()
    counts: dict[str, dict[str, int]] = {}
    waiting, problems = [], []
    for t in load_manifest():
        status, _, _ = check_take(t, pieces)
        kind = ("ready" if status == "ok" else "excluded" if status.startswith("excluded")
                else "waiting" if status == WAITING else "problem")
        c = counts.setdefault(t.piece, {"ready": 0, "waiting": 0, "excluded": 0, "problem": 0})
        c[kind] += 1
        if kind == "waiting":
            waiting.append(t.id)
        elif kind == "problem":
            problems.append(f"{t.id}: {status}")
    print(f"{'piece':<18}{'ready':>7}{'waiting':>9}{'excluded':>10}{'problem':>9}")
    total = {k: 0 for k in ("ready", "waiting", "excluded", "problem")}
    for piece in sorted(counts):
        c = counts[piece]
        print(f"{piece:<18}{c['ready']:>7}{c['waiting']:>9}{c['excluded']:>10}{c['problem']:>9}")
        for k in total:
            total[k] += c[k]
    print(f"{'TOTAL':<18}{total['ready']:>7}{total['waiting']:>9}{total['excluded']:>10}{total['problem']:>9}")
    if waiting:
        print(f"\nWaiting for labels: {', '.join(waiting)}\n"
              f"  (no labeling rows yet? python -m eval.cli sheet-template --new --out eval/sheets/to_label.tsv)")
    for p in problems:
        print(f"PROBLEM {p}")
    return 0 if not problems else 1


def cmd_evaluate(args) -> int:
    """sync -> import-sheet -> status -> run: the whole loop after recording or labeling."""
    print("== 1. New recordings")
    cmd_sync(argparse.Namespace(dry_run=False, split="dev"))
    if args.sheet:
        print(f"\n== 2. Labels from {Path(args.sheet).name}")
        if cmd_import_sheet(argparse.Namespace(sheet=args.sheet)):
            return 1
    else:
        print("\n== 2. Labels: no --sheet given, using the label files already imported")
    print("\n== 3. Status")
    cmd_status(args)
    print("\n== 4. Evaluation")
    rc = cmd_run(args)
    if rc == 0:
        print("Explore it in eval/results.ipynb (it opens the latest run).")
    return rc


SHEET_COLUMNS = ("take", "piece", "note_id", "pitch", "measure", "beat",
                 "label", "played", "flag", "comment")


def cmd_sheet_template(args) -> int:
    """One row per score note per take, tab-separated so it pastes straight into Google Sheets.

    Edit only label / played / flag / comment:
      label   todo (until you've listened) -> correct | wrong_pitch | missed | extra
      played  pitch actually played, e.g. E4 (wrong_pitch / extra; optional)
      flag    held_long | early | late (optional, any label)
    An extra note gets its own new row: note_id = the score note it follows
    (or START), label = extra.
    """
    pieces = all_pieces()
    takes = load_manifest()
    takes = [t for t in takes if not t.exclude]
    if args.take:
        takes = [t for t in takes if t.id in args.take]
    if args.piece:
        takes = [t for t in takes if t.piece in args.piece]
    if args.new:
        takes = [t for t in takes if not t.labels_path.exists()]
    text = sheet_rows(takes, pieces)
    if args.out:
        Path(args.out).write_text(text)
        print(f"Wrote {text.count(chr(10)) - 1} rows for {len(takes)} takes to {args.out}")
    else:
        sys.stdout.write(text)
    return 0


def sheet_rows(takes, pieces, header: bool = True) -> str:
    """Tab-separated rows, label = todo, so a take pasted but not yet listened to is never imported."""
    lines = ["\t".join(SHEET_COLUMNS)] if header else []
    for t in takes:
        spec = pieces[t.piece]
        if spec.pending:
            print(f"skip {t.id}: piece pending", file=sys.stderr)
            continue
        for n in spec.notes():
            lines.append("\t".join([t.id, t.piece, n.id, n.name, str(n.measure),
                                    f"{n.beat_in_measure:g}", "todo", "", "", ""]))
    return "\n".join(lines) + "\n"


def cmd_import_sheet(args) -> int:
    from eval.sheet import parse_sheet, write_label_files
    path = Path(args.sheet)
    res = parse_sheet(path, load_manifest(), all_pieces())
    for w in res.warnings:
        print(f"WARN  {w}")
    for s in res.skipped:
        print(f"SKIP  {s}: excluded in manifest")
    if res.todo:
        print(f"WAIT  {len(res.todo)} takes still have label = todo, not imported: {', '.join(res.todo)}")
    if res.problems:
        for p in res.problems:
            print(f"ERROR {p}")
        print(f"\n{len(res.problems)} problems; no label files written. Fix the sheet and re-import.")
        return 1
    written = write_label_files(res, path.name)
    n_perfect = sum(1 for st in res.takes.values() if not st.errors)
    print(f"\nWrote {len(written)} label files ({n_perfect} perfect, {len(written) - n_perfect} with errors) "
          f"under {Path('eval/labels')}")
    return 0


WAITING = "waiting for labels"


def check_take(take, pieces) -> tuple[str, object, object]:
    """Returns (status, spec, labels); status is 'ok' or a reason the take can't be scored."""
    if take.exclude:
        return f"excluded: {take.exclude}", None, None
    spec = pieces.get(take.piece)
    if spec is None:
        return f"unknown piece '{take.piece}'", None, None
    if spec.pending:
        return f"piece '{take.piece}' is pending transcription", spec, None
    if not take.audio.exists():
        return f"audio missing: {take.audio.relative_to(REPO_DIR)}", spec, None
    if not take.labels_path.exists():
        return WAITING, spec, None
    try:
        labels = load_labels(take.labels_path, spec)
    except (ValueError, KeyError) as e:
        return f"bad labels: {e}", spec, None
    if labels.todo:
        return WAITING, spec, labels
    return "ok", spec, labels


def cmd_validate(args) -> int:
    pieces = all_pieces()
    for name, spec in pieces.items():
        if spec.pending:
            print(f"WARN  piece {name}: pending transcription")
    takes = load_manifest()
    n_ok, n_bad, n_excluded, n_wait = 0, 0, 0, 0
    for t in takes:
        status, _, _ = check_take(t, pieces)
        if status == "ok":
            n_ok += 1
            if args.verbose:
                print(f"OK    {t.id} ({t.piece}, {t.split})")
        elif status.startswith("excluded"):
            n_excluded += 1
            print(f"SKIP  {t.id} ({t.piece}): {status}")
        elif status == WAITING:
            n_wait += 1
            print(f"WAIT  {t.id} ({t.piece}): {status}")
        else:
            n_bad += 1
            print(f"ERROR {t.id} ({t.piece}): {status}")
    by_split = {s: sum(1 for t in takes if t.split == s and not t.exclude) for s in SPLITS}
    print(f"\n{len(takes)} takes ({', '.join(f'{k}={v}' for k, v in by_split.items())}, "
          f"excluded={n_excluded}); {n_ok} ready, {n_wait} waiting for labels, {n_bad} with problems")
    return 0 if n_bad == 0 else 1


# ---------------------------------------------------------------- run
def cmd_run(args) -> int:
    from eval.adapters import get_adapter
    from eval.metrics import evaluate_take, summarize
    from eval.runs import new_run_dir, print_summary, print_take_errors, write_json, write_rows

    adapter = get_adapter(args.model)
    pieces = all_pieces()
    takes = load_manifest()
    if args.split != "all":
        takes = [t for t in takes if t.split == args.split]
    if args.piece:
        takes = [t for t in takes if t.piece in args.piece]
    if args.take:
        takes = [t for t in takes if t.id in args.take]

    ready, waiting = [], []
    for t in takes:
        status, spec, labels = check_take(t, pieces)
        if status == "ok":
            ready.append((t, spec, labels))
        elif status == WAITING:
            waiting.append(t.id)
        elif not status.startswith("excluded"):
            print(f"skip  {t.id}: {status}")
    if waiting:
        print(f"skip  {len(waiting)} takes waiting for labels: {', '.join(waiting)}")
    if not ready:
        print("Nothing to evaluate.")
        return 1

    run_dir, git = new_run_dir(adapter.name)
    print(f"Run dir: {run_dir.relative_to(REPO_DIR)}  ({len(ready)} takes, split={args.split})")
    evals, predictions, failures = [], {}, {}
    for t, spec, labels in ready:
        t0 = time.time()
        try:
            pred = adapter.predict(str(model_audio(t.audio)), spec, run_dir / "takes" / t.id)
            pred.validate([n.id for n in spec.notes()])
            predictions[t.id] = pred.to_json()
            status = f"{time.time() - t0:5.1f}s"
        except Exception:
            pred = None
            failures[t.id] = traceback.format_exc()
            status = "FAILED"
        ev = evaluate_take(t.id, spec, labels, pred)
        evals.append(ev)
        print(f"  {status}  {t.id:<28} {spec.name:<14} gold errors={ev.gold_errors} predicted={ev.pred_errors}")

    metrics = summarize(evals)
    rows = [r for ev in evals for r in ev.rows]
    write_json(run_dir / "meta.json", {
        "model": adapter.name, "model_config": adapter.config(), "git": git,
        "split": args.split, "argv": sys.argv[1:],
        "takes": [{"id": t.id, "piece": t.piece, "split": t.split,
                   "audio": str(t.audio.relative_to(REPO_DIR))} for t, _, _ in ready],
        "failures": failures,
    })
    write_json(run_dir / "predictions.json", predictions)
    write_json(run_dir / "metrics.json", metrics)
    write_rows(run_dir / "per_note.csv", rows)
    for tid, tb in failures.items():
        print(f"\n--- {tid} failed ---\n{tb}")
    print_summary(metrics, title=f"{adapter.name} @ {git['commit']}{' (dirty)' if git['dirty'] else ''}"
                                 f"  split={args.split}")
    print_take_errors(rows)
    try:
        from eval.plots import plot_run
        plot_run(run_dir)
        print(f"Charts: {(run_dir / 'plots').relative_to(REPO_DIR)}/")
    except Exception as e:                      # charts are a convenience; never fail the run
        print(f"Charts not generated: {e}")
    print(f"\nSaved to {run_dir.relative_to(REPO_DIR)}")
    return 0


def cmd_plots(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from eval.plots import plot_run
    from eval.runs import resolve_run
    for path in plot_run(resolve_run(args.run)):
        print(path.relative_to(REPO_DIR) if path.is_relative_to(REPO_DIR) else path)
    return 0


def cmd_compare(args) -> int:
    from eval.runs import compare_runs, resolve_run
    compare_runs(resolve_run(args.run_a), resolve_run(args.run_b))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("evaluate", help="sync new recordings, import the sheet, run, chart")
    p.add_argument("--sheet", help="labeling sheet exported from Google Sheets (.csv/.tsv)")
    p.add_argument("--model", default="v2")
    p.add_argument("--split", choices=SPLITS + ("all",), default="dev",
                   help="which takes to run (default dev)")
    p.add_argument("--piece", nargs="+")
    p.add_argument("--take", nargs="+")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("sync", help="register new recordings found in the intake folders")
    p.add_argument("--dry-run", action="store_true", help="only show what would be added")
    p.add_argument("--split", choices=SPLITS, default="dev")
    p.set_defaults(func=lambda a: (cmd_sync(a), 0)[1])

    p = sub.add_parser("status", help="per piece: takes ready, waiting for labels, excluded")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("convert", help="convert m4a recordings to wav")
    p.add_argument("files", nargs="*", help="m4a files (default: every .m4a take in the manifest)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("pieces", help="list piece specs")
    p.set_defaults(func=cmd_pieces)

    p = sub.add_parser("score", help="print a piece's numbered notes")
    p.add_argument("piece")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("new-take", help="register a recording and write its label template")
    p.add_argument("audio")
    p.add_argument("--piece", required=True)
    p.add_argument("--split", choices=SPLITS, default="dev")
    p.add_argument("--id", help="take id (default: audio file name)")
    p.set_defaults(func=cmd_new_take)

    p = sub.add_parser("sheet-template", help="per-note labeling sheet (TSV) for Google Sheets")
    p.add_argument("--out", help="output .tsv (default: stdout)")
    p.add_argument("--new", action="store_true", help="only takes without a label file yet")
    p.add_argument("--piece", nargs="+")
    p.add_argument("--take", nargs="+")
    p.set_defaults(func=cmd_sheet_template)

    p = sub.add_parser("import-sheet", help="write label files from the exported labeling sheet")
    p.add_argument("sheet", help=".csv or .tsv exported from Google Sheets")
    p.set_defaults(func=cmd_import_sheet)

    p = sub.add_parser("validate", help="check manifest, audio and labels")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("run", help="run a model over the manifest and score it")
    p.add_argument("--model", default="v2")
    p.add_argument("--split", choices=SPLITS + ("all",), default="dev",
                   help="default dev; the test split must be requested explicitly")
    p.add_argument("--piece", nargs="+")
    p.add_argument("--take", nargs="+")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("plots", help="draw charts for a run (also done automatically by run)")
    p.add_argument("run")
    p.set_defaults(func=cmd_plots)

    p = sub.add_parser("compare", help="compare two runs")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.set_defaults(func=cmd_compare)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
