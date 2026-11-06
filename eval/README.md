# Evaluation: adding recordings and scoring the model

All commands run from the repo root with the project venv:
`.venv/bin/python -m eval.cli <command>`.

## The loop

**1. Record.** Name each file `<player><piece>-<hands>-<take>.m4a` and drop it into
`recordings/eval-interviews/raw/`. Examples:

| file | piece |
|---|---|
| `3cmajor-1h-up.m4a` (or `3cmajor-up`) | C major scale, right hand |
| `3cmajor-2h-up.m4a` | C major scale, both hands |
| `3emajor-1h-down.m4a` | E major scale descending, right hand |
| `3ode-1h-2.m4a` / `3ode-2h-2.m4a` | Ode to Joy melody / with harmony |
| `3can-1h-1.m4a` / `3can-2h-1.m4a` | Canon in D left hand / both hands |

Anything after the hands part (`inc`, `miss`, `2`, ...) is free text. No spaces in names.
The rules live in [intake.toml](intake.toml); add a rule there when a new piece spec arrives.

**2. Register.**
```bash
.venv/bin/python -m eval.cli sync
```
This adds each new file to `manifest.toml` with the piece worked out from its name, converts it
to wav, and writes `eval/sheets/to_label_<date>.tsv`: one row per score note, with label = `todo`.
Files it can't place are listed with the reason and left alone. Run `sync --dry-run` first to preview.

**3. Label.** Paste the TSV rows (without the header row) below the existing rows of the shared
Google Sheet. For each take you listen to, change every `todo` to `correct`, then mark the mistakes
(`wrong_pitch` + played pitch, `missed`, an `extra` row, a flag). A take with any `todo` left is
not imported, so a half-done sheet never turns an unlabeled take into a "perfect" one.

**4. Evaluate.** Export the sheet as CSV, then:
```bash
.venv/bin/python -m eval.cli evaluate --sheet ~/Downloads/labels.csv
```
This syncs any newer recordings, imports the labels, prints a status table, runs the model on
every labeled take, and saves the run with charts. Then open `eval/results.ipynb`, which shows the
latest run, including the single-hand section.

Without `--sheet`, `evaluate` uses the labels already imported.

## Handy extras
| command | does |
|---|---|
| `status` | per piece: takes ready, waiting for labels, excluded |
| `sheet-template --new --out F.tsv` | labeling rows for every take that has no labels yet |
| `run --piece ode_melody` / `--take 1ode-1h-2` | evaluate a subset |
| `compare <runA> <runB>` | what changed between two runs (after a model change) |

To drop a bad take (e.g. the player restarted), add `exclude = "reason"` to its entry in
`manifest.toml`. It is then skipped everywhere but stays on record.
