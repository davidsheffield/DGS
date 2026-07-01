"""Preference-learning server for the Maker's Mark eigenshape space.

Where ``server.py`` ranks fixed PNGs and ``evolve_server.py`` breeds a grid you
hand-pick from, this serves a **forced A/B duel between marks generated from the
eigenspace** (``eigen.py``) and learns, with a Bayesian *peaked* model
(``preference_model.py``), what values along each eigen-axis you prefer.  Each
duel is chosen actively (dueling Thompson sampling) and logged for analysis by
``preference_display.py``.

Same stack as the other two apps: stdlib-only ThreadingHTTPServer on localhost.

    python3 preference_server.py [--port 8002] [--debug] [--samples DIR]
                                 [--data-dir DIR] [--var-keep 1.0]

A **session** lives in ``--data-dir`` (default ``pref_data/``): ``session.json``
pins the fitted ``PCABasis`` (so logged coefficients decode identically forever,
like an evolver run) and ``votes.jsonl`` is the append-only log.  On startup the
session is resumed if present and layout-compatible, and the preference model is
rebuilt from its votes.

API:
    GET  /                       the duel UI
    GET  /api/status             sizes, vote count, session id, n_components
    GET  /api/next?size=<bucket> issue a duel (two eigenspace marks)
    POST /api/vote               {"duel_id", "winner":"a"|"b"|"tie", "size"}
                                 -> log it, update the model, return the next duel
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from eigen import PCABasis
from genome import load_samples
from preference_model import WINNER_Y, PreferenceModel

ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "Samples"
DATA_DIR = ROOT / "pref_data"
STATIC_DIR = ROOT

HOST = "127.0.0.1"
PORT = 8002
DEBUG = False
VAR_KEEP = 1.0

SESSION_VERSION = 1
SIZES = ["20px", "30px", "40px", "50px", "60px", "70px", "80px"]
DEFAULT_SIZE = "40px"
MAX_OUTSTANDING = 1000          # cap on un-voted issued duels held in memory

RNG = random.Random()
LOCK = threading.Lock()         # guards MODEL, DUELS and the votes.jsonl append

# Session globals, populated by init_session().
SESSION: dict = {}
BASIS: PCABasis | None = None
MODEL: PreferenceModel | None = None
DUELS: "OrderedDict[str, tuple]" = OrderedDict()   # duel_id -> (a_coeffs, b_coeffs)


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


# ---------------------------------------------------------------------------
# Session + log persistence
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _session_path() -> Path:
    return DATA_DIR / "session.json"


def _votes_path() -> Path:
    return DATA_DIR / "votes.jsonl"


def save_session() -> None:
    tmp = DATA_DIR / "session.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(SESSION, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, _session_path())


def load_votes(session_id: str) -> list[dict]:
    """Logged votes for this session, in order, ready for observe_many()."""
    path = _votes_path()
    if not path.is_file():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (rec.get("session") == session_id and rec.get("winner") in WINNER_Y
                    and isinstance(rec.get("a_coeffs"), list)
                    and isinstance(rec.get("b_coeffs"), list)):
                out.append(rec)
    return out


def append_vote(record: dict) -> None:
    line = json.dumps(record, separators=(",", ":")) + "\n"
    with open(_votes_path(), "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def init_session() -> None:
    """Resume a compatible session or create a fresh one; build the model."""
    global SESSION, BASIS, MODEL
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if _session_path().is_file():
        state = json.loads(_session_path().read_text(encoding="utf-8"))
        try:
            basis = PCABasis.from_dict(state["basis"])       # raises on layout drift
        except ValueError as e:
            sys.exit(f"{_session_path()} is incompatible with this code ({e}).\n"
                     f"Move or delete {DATA_DIR} to start a fresh session.")
        SESSION, BASIS = state, basis
        votes = load_votes(state["session_id"])
        MODEL = PreferenceModel(basis.stds, rng=RNG)
        MODEL.observe_many(votes)
        print(f"resumed session {state['session_id'][:8]} — "
              f"{basis.n_components} components, {len(votes)} votes replayed")
        return

    genomes = load_samples(str(SAMPLES_DIR / "vector_*.svg"))
    if len(genomes) < 2:
        sys.exit(f"need at least 2 seed SVGs in {SAMPLES_DIR}, found {len(genomes)}")
    basis = PCABasis.fit(genomes, var_keep=VAR_KEEP)
    SESSION = {
        "version": SESSION_VERSION,
        "session_id": uuid.uuid4().hex,
        "created": _now(),
        "sizes": SIZES,
        "default_size": DEFAULT_SIZE,
        "basis": basis.to_dict(),
    }
    BASIS = basis
    MODEL = PreferenceModel(basis.stds, rng=RNG)
    save_session()
    print(f"new session {SESSION['session_id'][:8]} — fitted {basis.n_components} "
          f"components from {basis.n_seeds} seeds")


# ---------------------------------------------------------------------------
# Duels
# ---------------------------------------------------------------------------

def issue_duel(size: str) -> dict:
    """Pick the next duel (under LOCK), then decode its two marks to SVG."""
    with LOCK:
        a_coeffs, b_coeffs = MODEL.next_duel(RNG)
        duel_id = uuid.uuid4().hex
        DUELS[duel_id] = (a_coeffs, b_coeffs)
        while len(DUELS) > MAX_OUTSTANDING:
            DUELS.popitem(last=False)
        n_votes = MODEL.n_obs
    return {
        "duel_id": duel_id,
        "size": size,
        "n_votes": n_votes,
        "a": {"svg": BASIS.decode(a_coeffs).to_svg()},
        "b": {"svg": BASIS.decode(b_coeffs).to_svg()},
    }


def record_vote(duel_id, winner, size) -> dict:
    if winner not in WINNER_Y:
        raise ApiError(400, 'winner must be "a", "b" or "tie"')
    if size not in SIZES:
        raise ApiError(400, f"size must be one of {SIZES}")
    with LOCK:
        coeffs = DUELS.pop(duel_id, None)
        if coeffs is None:
            raise ApiError(400, "unknown or expired duel_id")
        a_coeffs, b_coeffs = coeffs
        append_vote({
            "ts": _now(),
            "session": SESSION["session_id"],
            "duel_id": duel_id,
            "a_coeffs": a_coeffs,
            "b_coeffs": b_coeffs,
            "size": size,
            "winner": winner,
            "n_components": BASIS.n_components,
        })
        MODEL.observe(a_coeffs, b_coeffs, winner)
    return issue_duel(size)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

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

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ApiError(400, "invalid json")
        if not isinstance(body, dict):
            raise ApiError(400, "body must be a json object")
        return body

    def _query_size(self, path_with_query: str) -> str:
        q = path_with_query.split("?", 1)
        if len(q) == 2:
            for part in q[1].split("&"):
                if part.startswith("size="):
                    size = part[len("size="):]
                    if size not in SIZES:
                        raise ApiError(400, f"size must be one of {SIZES}")
                    return size
        return DEFAULT_SIZE

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path == "/":
                self._send_file(STATIC_DIR / "preference.html", "text/html; charset=utf-8")
            elif path == "/static/preference.js":
                self._send_file(STATIC_DIR / "preference.js",
                                "application/javascript; charset=utf-8")
            elif path == "/static/preference.css":
                self._send_file(STATIC_DIR / "preference.css", "text/css; charset=utf-8")
            elif path == "/api/status":
                with LOCK:
                    n_votes = MODEL.n_obs
                self._send_json(HTTPStatus.OK, {
                    "session": SESSION["session_id"],
                    "sizes": SIZES,
                    "default_size": DEFAULT_SIZE,
                    "n_components": BASIS.n_components,
                    "n_votes": n_votes,
                })
            elif path == "/api/next":
                self._send_json(HTTPStatus.OK, issue_duel(self._query_size(self.path)))
            else:
                self._not_found()
        except ApiError as e:
            self._send_json(e.status, {"error": e.message})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/vote":
                body = self._read_body()
                payload = record_vote(body.get("duel_id"), body.get("winner"),
                                      body.get("size"))
                self._send_json(HTTPStatus.OK, payload)
            else:
                self._not_found()
        except ApiError as e:
            self._send_json(e.status, {"error": e.message})


def main() -> None:
    global DEBUG, DATA_DIR, SAMPLES_DIR, VAR_KEEP, PORT
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--debug", action="store_true", help="log every request")
    parser.add_argument("--samples", default=str(SAMPLES_DIR),
                        help="seed SVG directory (used when creating a session)")
    parser.add_argument("--data-dir", default=str(DATA_DIR),
                        help="where session.json + votes.jsonl live (default pref_data/)")
    parser.add_argument("--var-keep", type=float, default=VAR_KEEP,
                        help="fraction of variance to keep when fitting a new basis")
    args = parser.parse_args()
    DEBUG = args.debug
    DATA_DIR = Path(args.data_dir).expanduser().resolve()
    SAMPLES_DIR = Path(args.samples).expanduser().resolve()
    VAR_KEEP = args.var_keep
    PORT = args.port

    init_session()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    mode = " [debug]" if DEBUG else ""
    print(f"serving on http://{HOST}:{PORT} — data in {DATA_DIR}{mode}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.server_close()


if __name__ == "__main__":
    main()
