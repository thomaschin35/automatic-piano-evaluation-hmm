# Piano grader: local demo UI

Upload sheet music (PDF) and a recording; the page reads the notes, runs `stft_hmm_v3`, and marks
every note (correct / wrong note / missed / late / held too long), plus extra notes and places where the player
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
| `score_reader.py` (repo root) | PDF / image -> MusicXML (homr) -> the `PieceSpec` the algorithm follows, chords included. Also reads MusicXML and piece `.toml` files directly. |
| `ui/server.py` | Standard-library HTTP server: serves the page and the `/api/*` calls. |
| `ui/index.html` | The page (no build step, no dependencies). |

## Recording in the page

"Record" in the Recording card records from the computer's microphone, with echo cancellation, noise
suppression and auto gain turned off (they reshape piano sound). The level meter turns red when the
input is too loud. Stop uploads the take like a file (WebM or M4A, converted with ffmpeg). Your phone
is still the audio the algorithm was tuned on; a laptop mic is a different test.

## Going back (restarts)

When the player goes back and plays a passage again, every attempt is graded, not only the last:

- **Your take, start to finish**: the recording as one strip, one block per attempt (striped =
  abandoned), a tick per note (red = wrong), and where the player went back.
- **Final grade** tab: every attempt in the order played, one under the other, each with its
  own header (time, bars, score, and "went back to ..." or "played to the end"). The percentage at
  the top judges each note by the last attempt that reached it.
- **Attempt** tabs: one attempt alone, with its score and mistakes.
- Each attempt shows only the bars it played, plus one faded bar after the point where it stopped;
  the last attempt runs to the end of the piece. A "went back" badge marks where an attempt stopped.
- The outlined note follows playback through every attempt, and the bar inspector shows the bar
  as played in the attempt you clicked.

`grade.py` returns this as `attempts` (one entry per attempt with its notes and summary); each
final note carries the `attempt` that graded it, and each extra note the attempt it was played in.

## Inspecting a bar

Below the results, **Inside bar N** shows everything the algorithm saw in one bar (click a bar
number or a note to pick it; it opens on the first bar with a mistake). All panels share one time
axis, with the beats dashed and every detected strike as a grey line:

- waveform, with the written notes placed where the alignment put them (red = wrong, with the key heard)
- loudness and broadband onset
- three key-by-time maps: the raw spectrum level at each key's fundamental (before any processing),
  the key's presence after overtone removal (what v3 hears as sounding), and its onset score
- selected keys as lines (presence, onset score or raw level), to compare e.g. E3 against E4
- one card per strike: the written note it was matched to (and the verdict), the keys it was heard
  as, the keys set aside as an overtone of or spill from another key, and the raw evidence

Hover for exact values; click to play from that point to the end of the bar ("Play this bar" plays all of it). The data comes from `ui/measure_view.py`;
the server keeps the last 4 grading runs in memory for this.

## Things to know

- **Check the notes before grading.** After a PDF is read, the notes appear in editable boxes
  (`E4:1 F#4:0.5 rest:2`, one beat = a quarter note). A misread note would be graded as a wrong
  note, so fix it there first. MusicXML uploads and the built-in pieces skip the recognition step.
- **Chords within one hand** are kept: notes that start together on a staff are one chord, and
  each of its keys is graded on its own. Type one as `C4+E4+G4:2`. The algorithm accepts chords but
  has not been tuned or tested on them yet, so expect weaker results there than on single notes:
  a chord rolled over more than about 70 ms reads as missed + extra, a wrong extra key inside a
  chord is reported as an extra rather than a wrong note, and held too long is not checked on chords.
- **The score shown is "right and on time"**: notes that were the written key and not late, out of
  all written notes. Extra notes, held-too-long notes and going back are listed but do not change
  the percentage (held too long is detected conservatively and is not in the algorithm's goal score
  yet, see the main README).
- Repeats, voltas and grace notes in the sheet music are not expanded.
