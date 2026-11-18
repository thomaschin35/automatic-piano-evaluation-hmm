# Piano grader: local demo UI

Upload sheet music (PDF) and a recording; the page reads the notes, runs `stft_hmm_v3`, and marks
every note (correct / wrong note / missed / late), plus extra notes and places where the player
went back.

## Run

From the repo root:

```bash
.venv/bin/python ui/server.py
```

Then open http://127.0.0.1:8000. `--port 8001` picks another port. Nothing leaves the machine;
uploads live in a temp folder that is deleted when the server stops.

## One-time setup for reading PDFs

The PDF reader is [homr](https://github.com/liebharc/homr) (optical music recognition). It lives in
its own virtual env so its dependencies never touch the algorithm's env:

```bash
python3 -m venv .venv-omr
```

```bash
.venv-omr/bin/pip install homr
```

```bash
.venv-omr/bin/homr --init
```

The last command downloads the recognition models once. Reading a page takes roughly 1-2 minutes
on a laptop CPU. `.m4a` recordings need `ffmpeg` on the PATH; `.wav` and `.mp3` do not.

## Files

| File | What it does |
|---|---|
| `grade.py` (repo root) | One entry point for the algorithm: score + recording in, JSON-ready result out. Also a command line: `grade.py --audio take.wav --piece ode_melody` or `--score sheet.pdf`. |
| `score_reader.py` (repo root) | PDF / image -> MusicXML (homr) -> the `PieceSpec` the algorithm follows. Also reads MusicXML and piece `.toml` files directly. |
| `ui/server.py` | Standard-library HTTP server: serves the page and the `/api/*` calls. |
| `ui/index.html` | The page (no build step, no dependencies). |

## Things to know

- **Check the notes before grading.** After a PDF is read, the notes appear in editable boxes
  (`E4:1 F#4:0.5 rest:2`, one beat = a quarter note). A misread note would be graded as a wrong
  note, so fix it there first. MusicXML uploads and the built-in pieces skip the recognition step.
- **One note at a time per hand.** Chords inside a hand are reduced to one note (top note in the
  right hand, bottom in the left) with a warning. The reduction is one function,
  `score_reader._line`, which is the place to change when the algorithm learns chords.
- **The score shown is "right and on time"**: notes that were the written key and not late, out of
  all written notes. Extra notes and going back are listed but do not change the percentage.
- Repeats, voltas and grace notes in the sheet music are not expanded.
