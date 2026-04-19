import argparse
import json
import os
import random
import sys
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "Samples"
STATIC_DIR = ROOT
VOTES_PATH = ROOT / "votes.jsonl"

SIZES = ["20px", "30px", "40px", "50px", "60px", "70px", "80px"]
HOST = "127.0.0.1"
PORT = 8000
DEBUG = False

IMAGES: list[str] = sorted(
    p.name for p in SAMPLES_DIR.iterdir()
    if p.is_file() and p.suffix.lower() == ".png"
)


def append_vote(record: dict) -> None:
    line = json.dumps(record, separators=(",", ":")) + "\n"
    with open(VOTES_PATH, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        if DEBUG:
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _not_found(self) -> None:
        self.send_response(HTTPStatus.NOT_FOUND)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]

        if path == "/":
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return

        if path == "/static/app.js":
            self._send_file(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")
            return

        if path == "/static/style.css":
            self._send_file(STATIC_DIR / "style.css", "text/css; charset=utf-8")
            return

        if path.startswith("/samples/"):
            name = path[len("/samples/"):]
            if name in IMAGES:
                self._send_file(SAMPLES_DIR / name, "image/png")
                return
            self._not_found()
            return

        if path == "/api/pair":
            a, b = random.sample(IMAGES, 2)
            self._send_json(HTTPStatus.OK, {
                "pair_id": uuid.uuid4().hex,
                "a": a,
                "b": b,
                "size": random.choice(SIZES),
            })
            return

        self._not_found()

    def do_POST(self):
        if self.path != "/api/vote":
            self._not_found()
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid json"})
            return

        pair_id = body.get("pair_id")
        a = body.get("a")
        b = body.get("b")
        size = body.get("size")
        winner = body.get("winner")

        if (
            not isinstance(pair_id, str)
            or a not in IMAGES
            or b not in IMAGES
            or size not in SIZES
            or winner not in ("a", "b")
        ):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid vote"})
            return

        append_vote({
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pair_id": pair_id,
            "a": a,
            "b": b,
            "size": size,
            "winner": winner,
        })
        self._send_json(HTTPStatus.OK, {"ok": True})


def main() -> None:
    global DEBUG
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="log every request")
    args = parser.parse_args()
    DEBUG = args.debug

    if len(IMAGES) < 2:
        sys.exit(f"need at least 2 PNGs in {SAMPLES_DIR}, found {len(IMAGES)}")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    mode = " [debug]" if DEBUG else ""
    print(f"serving on http://{HOST}:{PORT} — {len(IMAGES)} images loaded{mode}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.server_close()


if __name__ == "__main__":
    main()
