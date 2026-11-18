"""
ui/server.py
Local demo server (standard library only). From the repo root:

    .venv/bin/python ui/server.py            # then open http://127.0.0.1:8000

    GET  /                      the page (ui/index.html)
    GET  /api/pieces            built-in pieces from eval/pieces/
    GET  /api/piece?name=X      one built-in piece as a score
    POST /api/score?filename=F  body = sheet music file  -> {score, warnings, seconds}
    POST /api/audio?filename=F  body = recording         -> {audio_id}
    POST /api/grade             body = {audio_id, score} -> grade.to_result(...)

Uploads are kept in a temp folder that is deleted when the server stops.
"""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
import traceback
from urllib.parse import parse_qs, urlparse
import uuid

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import grade as grader                                              # noqa: E402
from score_reader import ScoreReadError, read_score, spec_from_dict, spec_to_dict   # noqa: E402
from eval.pieces import all_pieces                                  # noqa: E402

PAGE = Path(__file__).resolve().parent / "index.html"
MAX_UPLOAD = 200 * 1024 * 1024
SCORE_TYPES = {".pdf", ".png", ".jpg", ".jpeg", ".musicxml", ".xml", ".mxl"}
UPLOADS: Path                                   # set in main()
GRADE_LOCK = threading.Lock()                   # one recognition / grading job at a time


class UserError(Exception):
    """Shown to the user as-is (HTTP 400)."""


def _suffix(filename: str) -> str:
    m = re.search(r"\.[A-Za-z0-9]{1,8}$", filename or "")
    return m.group(0).lower() if m else ""


class Handler(BaseHTTPRequestHandler):
    server_version = "PianoGrader/1"

    def log_message(self, fmt, *args):
        sys.stderr.write("  %s\n" % (fmt % args))

    # ---- plumbing
    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json")

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            raise UserError("The upload was empty")
        if n > MAX_UPLOAD:
            raise UserError("The file is larger than 200 MB")
        return self.rfile.read(n)

    def _handle(self, fn) -> None:
        try:
            self._json(fn())
        except (UserError, ScoreReadError) as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:                                       # algorithm or server fault
            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    # ---- routes
    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/pieces":
            self._handle(lambda: [{"name": n, "title": s.title, "two_hands": not s.is_mono}
                                  for n, s in all_pieces().items() if not s.pending])
        elif url.path == "/api/piece":
            self._handle(lambda: self._piece(q.get("name", [""])[0]))
        else:
            self._json({"error": "Not found"}, 404)

    def do_POST(self):
        url = urlparse(self.path)
        filename = parse_qs(url.query).get("filename", [""])[0]
        if url.path == "/api/score":
            self._handle(lambda: self._score(filename))
        elif url.path == "/api/audio":
            self._handle(lambda: self._audio(filename))
        elif url.path == "/api/grade":
            self._handle(self._grade)
        else:
            self._json({"error": "Not found"}, 404)

    def _piece(self, name: str) -> dict:
        pieces = all_pieces()
        if name not in pieces or pieces[name].pending:
            raise UserError(f"Unknown piece '{name}'")
        return {"score": spec_to_dict(pieces[name]), "warnings": []}

    def _score(self, filename: str) -> dict:
        suffix = _suffix(filename)
        if suffix not in SCORE_TYPES:
            raise UserError("Upload the score as a PDF, an image (PNG / JPG) or MusicXML")
        path = UPLOADS / f"score_{uuid.uuid4().hex}{suffix}"
        path.write_bytes(self._body())
        t0 = time.time()
        try:
            with GRADE_LOCK:
                spec, warnings = read_score(path)
        finally:
            path.unlink(missing_ok=True)
        d = spec_to_dict(spec)
        if d["title"] == "Uploaded score" and filename:
            d["title"] = Path(filename).stem
        return {"score": d, "warnings": warnings, "seconds": round(time.time() - t0, 1)}

    def _audio(self, filename: str) -> dict:
        suffix = _suffix(filename) or ".wav"
        audio_id = uuid.uuid4().hex
        (UPLOADS / f"audio_{audio_id}{suffix}").write_bytes(self._body())
        return {"audio_id": audio_id}

    def _grade(self) -> dict:
        try:
            req = json.loads(self._body())
        except json.JSONDecodeError as e:
            raise UserError("Bad request") from e
        audio_id = str(req.get("audio_id", ""))
        if not re.fullmatch(r"[0-9a-f]{32}", audio_id):
            raise UserError("Upload a recording first")
        found = list(UPLOADS.glob(f"audio_{audio_id}.*"))
        if not found:
            raise UserError("The recording is no longer on the server; upload it again")
        spec = spec_from_dict(req.get("score") or {})
        t0 = time.time()
        with GRADE_LOCK:
            result = grader.grade(found[0], spec)
        result["seconds"] = round(time.time() - t0, 1)
        return result


def main() -> None:
    global UPLOADS
    ap = argparse.ArgumentParser(description="Local demo UI for the piano grader.")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="piano_grader_") as tmp:
        UPLOADS = Path(tmp)
        httpd = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
        print(f"Piano grader running at http://127.0.0.1:{a.port}  (Ctrl+C to stop)", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
