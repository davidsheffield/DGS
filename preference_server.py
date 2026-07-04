"""Preference-learning server for the Maker's Mark eigenshape space.

Where ``server.py`` ranks fixed PNGs and ``evolve_server.py`` breeds a grid you
hand-pick from, this serves a **forced A/B duel between marks generated from the
eigenspace** (``eigen.py``) and learns, with a Bayesian *peaked* model
(``preference_model.py``), what values along each eigen-axis you prefer.  Each
duel is chosen by a hybrid scheduler (axis staircase duels + seed-blend duels +
on-request confirm duels -- see ``preference_model.py``) and logged for analysis
by ``preference_display.py``.

Same stack as the other two apps: stdlib-only ThreadingHTTPServer on localhost.

    python3 preference_server.py [--port 8002] [--debug] [--samples DIR]
                                 [--data-dir DIR] [--var-keep 1.0]
                                 [--active-var 0.9]

A **session** lives in ``--data-dir`` (default ``pref_data/``): ``session.json``
pins the fitted ``PCABasis`` (so logged coefficients decode identically forever,
like an evolver run), the number of *active* axes (``n_active`` -- the leading
axes carrying ``--active-var`` of the population variance) and the seed marks'
standardized coefficients (``seed_zs``, used for blend-duel candidates).
``votes.jsonl`` is the append-only log.  On startup the session is resumed if
present and layout-compatible (old sessions missing ``n_active``/``seed_zs``
are migrated in place), and one preference model **per size bucket** is rebuilt
from that bucket's replayed votes.

API:
    GET  /                       the duel UI
    GET  /api/status             sizes, vote counts (overall + per size),
                                 session id, n_components, n_active
    GET  /api/next?size=<bucket>&mode=confirm   issue a duel (two eigenspace
                                 marks); ``mode`` is optional, and only
                                 "confirm" may be passed explicitly
    POST /api/vote               {"duel_id", "winner":"a"|"b"|"tie", "size",
                                 "mode"?} -> log it, update that size's model,
                                 return the next duel (honoring "mode" again)
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
ACTIVE_VAR = 0.9                # fraction of variance the active axes must carry

SESSION_VERSION = 1
SIZES = ["20px", "30px", "40px", "50px", "60px", "70px", "80px"]
DEFAULT_SIZE = "40px"
MAX_OUTSTANDING = 1000          # cap on un-voted issued duels held in memory

RNG = random.Random()
LOCK = threading.Lock()         # guards MODELS, DUELS and the votes.jsonl append

# Session globals, populated by init_session().
SESSION: dict = {}
BASIS: PCABasis | None = None
MODELS: "dict[str, PreferenceModel]" = {}          # size bucket -> its own model
DUELS: "OrderedDict[str, tuple]" = OrderedDict()   # duel_id -> (a_coeffs, b_coeffs, size, meta)


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


def compute_n_active(stds: list[float], active_var: float) -> int:
    """Smallest M whose cumulative variance share (stds are already sorted
    descending) reaches ``active_var``."""
    total = sum(s * s for s in stds) or 1.0
    cum = 0.0
    for i, s in enumerate(stds):
        cum += s * s
        if cum / total >= active_var:
            return i + 1
    return len(stds)


def seed_zs_for(basis: PCABasis, genomes) -> list[list[float]]:
    """Encode each seed genome with the fitted basis and standardize by
    ``stds`` -> z units, for use as blend-duel candidates."""
    return [[c / s for c, s in zip(basis.encode(g), basis.stds)] for g in genomes]


def get_model(size: str) -> PreferenceModel:
    """The (lazily created) per-size-bucket model, sharing the session's
    fitted basis / active-axis count / seed blends."""
    if size not in MODELS:
        MODELS[size] = PreferenceModel(
            BASIS.stds, seed_zs=SESSION.get("seed_zs"),
            n_active=SESSION.get("n_active"), rng=RNG)
    return MODELS[size]


def init_session() -> None:
    """Resume a compatible session or create a fresh one; build one model per
    size bucket from that bucket's replayed votes."""
    global SESSION, BASIS, MODELS
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if _session_path().is_file():
        state = json.loads(_session_path().read_text(encoding="utf-8"))
        try:
            basis = PCABasis.from_dict(state["basis"])       # raises on layout drift
        except ValueError as e:
            sys.exit(f"{_session_path()} is incompatible with this code ({e}).\n"
                     f"Move or delete {DATA_DIR} to start a fresh session.")
        SESSION, BASIS = state, basis

        migrated = False
        if "n_active" not in SESSION:
            SESSION["n_active"] = compute_n_active(basis.stds, ACTIVE_VAR)
            migrated = True
        if "seed_zs" not in SESSION:
            genomes = load_samples(str(SAMPLES_DIR / "vector_*.svg"))
            SESSION["seed_zs"] = seed_zs_for(basis, genomes)
            migrated = True
            print(f"note: old session had no seed_zs -- encoded the current "
                  f"Samples/ ({len(genomes)} seeds) with the pinned basis; "
                  f"blend duels use today's seeds, not the ones the basis was fit from")
        if migrated:
            save_session()

        votes = load_votes(state["session_id"])
        MODELS = {}
        by_size: dict[str, list[dict]] = {}
        for v in votes:
            by_size.setdefault(v.get("size", DEFAULT_SIZE), []).append(v)
        for size, vs in by_size.items():
            get_model(size).observe_many(vs)
        print(f"resumed session {state['session_id'][:8]} — "
              f"{basis.n_components} components ({SESSION['n_active']} active), "
              f"{len(votes)} votes replayed across {len(by_size)} size bucket(s)")
        return

    genomes = load_samples(str(SAMPLES_DIR / "vector_*.svg"))
    if len(genomes) < 2:
        sys.exit(f"need at least 2 seed SVGs in {SAMPLES_DIR}, found {len(genomes)}")
    basis = PCABasis.fit(genomes, var_keep=VAR_KEEP)
    n_active = compute_n_active(basis.stds, ACTIVE_VAR)
    SESSION = {
        "version": SESSION_VERSION,
        "session_id": uuid.uuid4().hex,
        "created": _now(),
        "sizes": SIZES,
        "default_size": DEFAULT_SIZE,
        "basis": basis.to_dict(),
        "n_active": n_active,
        "seed_zs": seed_zs_for(basis, genomes),
    }
    BASIS = basis
    MODELS = {}
    save_session()
    print(f"new session {SESSION['session_id'][:8]} — fitted {basis.n_components} "
          f"components ({n_active} active) from {basis.n_seeds} seeds")


# ---------------------------------------------------------------------------
# Duels
# ---------------------------------------------------------------------------

def issue_duel(size: str, mode: str | None = None) -> dict:
    """Pick the next duel (under LOCK) from ``size``'s model, then decode its
    two marks to SVG."""
    with LOCK:
        model = get_model(size)
        a_coeffs, b_coeffs, meta = model.next_duel(RNG, mode=mode)
        duel_id = uuid.uuid4().hex
        DUELS[duel_id] = (a_coeffs, b_coeffs, size, meta)
        while len(DUELS) > MAX_OUTSTANDING:
            DUELS.popitem(last=False)
        n_votes = model.n_obs
    return {
        "duel_id": duel_id,
        "size": size,
        "n_votes": n_votes,
        "mode": meta.get("mode"),
        "axis": meta.get("axis"),
        "a": {"svg": BASIS.decode(a_coeffs).to_svg()},
        "b": {"svg": BASIS.decode(b_coeffs).to_svg()},
    }


def record_vote(duel_id, winner, size, mode: str | None = None) -> dict:
    if winner not in WINNER_Y:
        raise ApiError(400, 'winner must be "a", "b" or "tie"')
    if size not in SIZES:
        raise ApiError(400, f"size must be one of {SIZES}")
    if mode is not None and mode != "confirm":
        raise ApiError(400, 'mode must be "confirm" if provided')
    with LOCK:
        entry = DUELS.get(duel_id)
        if entry is None:
            raise ApiError(400, "unknown or expired duel_id")
        a_coeffs, b_coeffs, issued_size, meta = entry
        if size != issued_size:
            # don't pop yet -- a rejected vote must leave the duel votable
            raise ApiError(400, f"size must match the issued duel ({issued_size!r})")
        del DUELS[duel_id]
        record = {
            "ts": _now(),
            "session": SESSION["session_id"],
            "duel_id": duel_id,
            "a_coeffs": a_coeffs,
            "b_coeffs": b_coeffs,
            "size": size,
            "winner": winner,
            "n_components": BASIS.n_components,
            "mode": meta.get("mode"),
        }
        if "axis" in meta:
            record["axis"] = meta["axis"]
        append_vote(record)
        get_model(size).observe(a_coeffs, b_coeffs, winner)
    return issue_duel(size, mode=mode)


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

    def _query_params(self, path_with_query: str) -> dict:
        q = path_with_query.split("?", 1)
        if len(q) < 2:
            return {}
        return dict(part.split("=", 1) for part in q[1].split("&") if "=" in part)

    def _query_size(self, params: dict) -> str:
        size = params.get("size", DEFAULT_SIZE)
        if size not in SIZES:
            raise ApiError(400, f"size must be one of {SIZES}")
        return size

    def _query_mode(self, params: dict) -> str | None:
        mode = params.get("mode")
        if mode is not None and mode != "confirm":
            raise ApiError(400, 'mode must be "confirm" if provided')
        return mode

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
                    votes_by_size = {size: m.n_obs for size, m in MODELS.items()}
                self._send_json(HTTPStatus.OK, {
                    "session": SESSION["session_id"],
                    "sizes": SIZES,
                    "default_size": DEFAULT_SIZE,
                    "n_components": BASIS.n_components,
                    "n_active": SESSION.get("n_active", BASIS.n_components),
                    "n_votes": sum(votes_by_size.values()),
                    "votes_by_size": votes_by_size,
                })
            elif path == "/api/next":
                params = self._query_params(self.path)
                size = self._query_size(params)
                mode = self._query_mode(params)
                self._send_json(HTTPStatus.OK, issue_duel(size, mode))
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
                                      body.get("size"), body.get("mode"))
                self._send_json(HTTPStatus.OK, payload)
            else:
                self._not_found()
        except ApiError as e:
            self._send_json(e.status, {"error": e.message})


def main() -> None:
    global DEBUG, DATA_DIR, SAMPLES_DIR, VAR_KEEP, ACTIVE_VAR, PORT
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--debug", action="store_true", help="log every request")
    parser.add_argument("--samples", default=str(SAMPLES_DIR),
                        help="seed SVG directory (used when creating a session)")
    parser.add_argument("--data-dir", default=str(DATA_DIR),
                        help="where session.json + votes.jsonl live (default pref_data/)")
    parser.add_argument("--var-keep", type=float, default=VAR_KEEP,
                        help="fraction of variance to keep when fitting a new basis")
    parser.add_argument("--active-var", type=float, default=ACTIVE_VAR,
                        help="fraction of variance the active (learned/varied) "
                             "axes must carry, when creating a new session")
    args = parser.parse_args()
    DEBUG = args.debug
    DATA_DIR = Path(args.data_dir).expanduser().resolve()
    SAMPLES_DIR = Path(args.samples).expanduser().resolve()
    VAR_KEEP = args.var_keep
    ACTIVE_VAR = args.active_var
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
