"""Interactive evolution server for the Maker's Mark eigenshape GA.

Serves a grid of candidate marks; the user selects the ones to propagate and
the server breeds the next generation in PCA coefficient space (see
``eigen.py``).  Runs live under ``--runs-dir`` (default ``runs/``), each as a
directory holding ``state.json`` -- everything needed to resume, including the
fitted PCA basis, so a resumed run decodes identically even if ``Samples/``
changes later -- plus one directory of rendered SVGs per generation.

Same stack as server.py: stdlib-only ThreadingHTTPServer bound to localhost.

    python3 evolve_server.py [--port 8001] [--debug] [--runs-dir DIR]
                             [--samples DIR] [--var-keep 1.0]

API:
    GET  /api/runs                     list resumable runs in runs-dir
    POST /api/runs                     {"name", "out_dir"?, "settings"?} -> new run
    GET  /api/runs/<name>              resume a run (latest generation)
    GET  /api/runs/<name>/gen/<g>      browse a past generation
    POST /api/runs/<name>/breed        {"selected", "n_offspring", "elitism"}
    POST /api/runs/<name>/restart      drop every generation after gen 0
    POST /api/runs/load                {"path"} -> load a run outside runs-dir
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from eigen import PCABasis, breed_coeffs, mutate_coeffs
from genome import load_samples

ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "Samples"
RUNS_DIR = ROOT / "runs"
STATIC_DIR = ROOT

HOST = "127.0.0.1"
PORT = 8001
DEBUG = False
VAR_KEEP = 1.0

STATE_VERSION = 1
RUN_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
MAX_OFFSPRING = 200

DEFAULT_SETTINGS = {
    "n_offspring": 16,
    "elitism": True,
    "blend_prob": 0.5,
    "alpha": 0.5,
    "mut_rate": 0.3,
    "mut_sigma": 0.35,
}

RNG = random.Random()


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class RunState:
    """A loaded run: parsed state.json + its basis + a breed/save lock."""

    def __init__(self, state: dict, basis: PCABasis, run_dir: Path):
        self.state = state
        self.basis = basis
        self.dir = run_dir
        self.lock = threading.Lock()


RUNS: dict[str, RunState] = {}          # name -> loaded run
RUNS_LOCK = threading.Lock()            # guards the RUNS dict itself


# ---------------------------------------------------------------------------
# Run persistence
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_state(run: RunState) -> None:
    """Atomic rewrite of state.json (tmp + rename)."""
    tmp = run.dir / "state.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(run.state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, run.dir / "state.json")


def write_generation_svgs(run: RunState, gen: dict) -> None:
    gen_dir = run.dir / f"gen_{gen['index']:03d}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    for cand in gen["candidates"]:
        svg = run.basis.decode(cand["coeffs"]).to_svg()
        (gen_dir / f"{cand['id']}.svg").write_text(svg, encoding="utf-8")


def create_run(name: str, out_dir: str | None, settings: dict) -> RunState:
    if not RUN_NAME_RE.fullmatch(name or ""):
        raise ApiError(400, "run name must match [A-Za-z0-9_-]{1,64}")
    base = Path(out_dir).expanduser() if out_dir else RUNS_DIR
    if not base.is_absolute():
        base = ROOT / base
    run_dir = base / name
    if (run_dir / "state.json").exists():
        raise ApiError(409, f"run already exists: {run_dir}")

    genomes = load_samples(str(SAMPLES_DIR / "vector_*.svg"))
    if len(genomes) < 2:
        raise ApiError(500, f"need at least 2 seed SVGs in {SAMPLES_DIR}, "
                            f"found {len(genomes)}")
    basis = PCABasis.fit(genomes, var_keep=VAR_KEEP)

    merged = dict(DEFAULT_SETTINGS)
    merged.update({k: settings[k] for k in DEFAULT_SETTINGS if k in settings})
    gen0 = {
        "index": 0,
        "candidates": [
            {"id": f"g0c{i:02d}", "coeffs": basis.encode(g),
             "parents": None, "origin": g.meta.get("origin", "seed")}
            for i, g in enumerate(genomes)
        ],
        "selected": [],
    }
    state = {
        "version": STATE_VERSION,
        "name": name,
        "created": _now(),
        "settings": merged,
        "basis": basis.to_dict(),
        "generations": [gen0],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    run = RunState(state, basis, run_dir)
    write_generation_svgs(run, gen0)
    save_state(run)
    return run


def load_run(run_dir: Path) -> RunState:
    state_path = run_dir / "state.json"
    if not state_path.is_file():
        raise ApiError(404, f"no state.json in {run_dir}")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        basis = PCABasis.from_dict(state["basis"])
    except (json.JSONDecodeError, KeyError) as e:
        raise ApiError(500, f"corrupt state.json in {run_dir}: {e}")
    except ValueError as e:                       # basis/layout version mismatch
        raise ApiError(409, str(e))
    return RunState(state, basis, run_dir)


def get_run(name: str) -> RunState:
    with RUNS_LOCK:
        run = RUNS.get(name)
    if run is not None:
        return run
    run = load_run(RUNS_DIR / name)
    with RUNS_LOCK:
        return RUNS.setdefault(name, run)


def list_runs() -> list[dict]:
    out = []
    if RUNS_DIR.is_dir():
        for state_path in sorted(RUNS_DIR.glob("*/state.json")):
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                out.append({
                    "name": state.get("name", state_path.parent.name),
                    "generations": len(state.get("generations", [])),
                    "updated": datetime.fromtimestamp(
                        state_path.stat().st_mtime, timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                })
            except (OSError, json.JSONDecodeError):
                continue
    return out


# ---------------------------------------------------------------------------
# Breeding
# ---------------------------------------------------------------------------

def breed_generation(run: RunState, selected: list[str], n_offspring: int,
                     elitism: bool) -> dict:
    with run.lock:
        gens = run.state["generations"]
        cur = gens[-1]
        by_id = {c["id"]: c for c in cur["candidates"]}

        if not selected:
            raise ApiError(400, "select at least one parent")
        bad = [s for s in selected if s not in by_id]
        if bad:
            raise ApiError(400, f"unknown candidate ids: {bad}")
        if not isinstance(n_offspring, int) or not 1 <= n_offspring <= MAX_OFFSPRING:
            raise ApiError(400, f"n_offspring must be an integer in 1..{MAX_OFFSPRING}")

        st = run.state["settings"]
        parents = [by_id[s] for s in selected]
        idx = cur["index"] + 1
        candidates = []
        for i in range(n_offspring):
            if len(parents) >= 2:
                pa, pb = RNG.sample(parents, 2)
                coeffs = breed_coeffs(
                    pa["coeffs"], pb["coeffs"], run.basis.stds,
                    blend_prob=st["blend_prob"], alpha=st["alpha"],
                    rate=st["mut_rate"], sigma=st["mut_sigma"], rng=RNG)
                pids = [pa["id"], pb["id"]]
            else:                        # single parent: mutation-only offspring
                pa = parents[0]
                coeffs = mutate_coeffs(pa["coeffs"], run.basis.stds,
                                       rate=st["mut_rate"], sigma=st["mut_sigma"],
                                       rng=RNG)
                pids = [pa["id"]]
            candidates.append({"id": f"g{idx}c{i:02d}", "coeffs": coeffs,
                               "parents": pids, "origin": "bred"})
        if elitism:
            for j, p in enumerate(parents):
                candidates.append({"id": f"g{idx}c{n_offspring + j:02d}",
                                   "coeffs": p["coeffs"],
                                   "parents": [p["id"]], "origin": "elite"})

        cur["selected"] = list(selected)
        st["n_offspring"] = n_offspring
        st["elitism"] = bool(elitism)
        new_gen = {"index": idx, "candidates": candidates, "selected": []}
        gens.append(new_gen)
        write_generation_svgs(run, new_gen)
        save_state(run)
        return new_gen


def restart_run(run: RunState) -> dict:
    with run.lock:
        gens = run.state["generations"]
        for gen in gens[1:]:
            shutil.rmtree(run.dir / f"gen_{gen['index']:03d}", ignore_errors=True)
        run.state["generations"] = gens[:1]
        run.state["generations"][0]["selected"] = []
        save_state(run)
        return run.state["generations"][0]


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------

def generation_payload(run: RunState, gen_index: int) -> dict:
    gens = run.state["generations"]
    if not 0 <= gen_index < len(gens):
        raise ApiError(404, f"generation {gen_index} does not exist")
    gen = gens[gen_index]
    return {
        "name": run.state["name"],
        "dir": str(run.dir),
        "generation": gen["index"],
        "n_generations": len(gens),
        "settings": run.state["settings"],
        "n_components": run.basis.n_components,
        "dim": run.basis.dim,
        "selected": gen["selected"],
        "candidates": [
            {"id": c["id"],
             "svg": run.basis.decode(c["coeffs"]).to_svg(),
             "parents": c["parents"],
             "origin": c["origin"]}
            for c in gen["candidates"]
        ],
    }


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

    # --- routing -----------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path == "/":
                self._send_file(STATIC_DIR / "evolve.html", "text/html; charset=utf-8")
            elif path == "/static/evolve.js":
                self._send_file(STATIC_DIR / "evolve.js",
                                "application/javascript; charset=utf-8")
            elif path == "/static/evolve.css":
                self._send_file(STATIC_DIR / "evolve.css", "text/css; charset=utf-8")
            elif path == "/api/runs":
                self._send_json(HTTPStatus.OK,
                                {"runs": list_runs(), "runs_dir": str(RUNS_DIR)})
            else:
                m = re.fullmatch(r"/api/runs/([A-Za-z0-9_-]{1,64})", path)
                if m:
                    run = get_run(m.group(1))
                    payload = generation_payload(
                        run, len(run.state["generations"]) - 1)
                    self._send_json(HTTPStatus.OK, payload)
                    return
                m = re.fullmatch(r"/api/runs/([A-Za-z0-9_-]{1,64})/gen/(\d+)", path)
                if m:
                    run = get_run(m.group(1))
                    self._send_json(HTTPStatus.OK,
                                    generation_payload(run, int(m.group(2))))
                    return
                self._not_found()
        except ApiError as e:
            self._send_json(e.status, {"error": e.message})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            body = self._read_body()
            if path == "/api/runs":
                run = create_run(body.get("name"), body.get("out_dir"),
                                 body.get("settings") or {})
                with RUNS_LOCK:
                    RUNS[run.state["name"]] = run
                self._send_json(HTTPStatus.OK, generation_payload(run, 0))
            elif path == "/api/runs/load":
                p = body.get("path")
                if not isinstance(p, str) or not p:
                    raise ApiError(400, "path required")
                run_dir = Path(p).expanduser()
                if not run_dir.is_absolute():
                    run_dir = ROOT / run_dir
                run = load_run(run_dir)
                with RUNS_LOCK:
                    RUNS[run.state["name"]] = run
                payload = generation_payload(run, len(run.state["generations"]) - 1)
                self._send_json(HTTPStatus.OK, payload)
            else:
                m = re.fullmatch(r"/api/runs/([A-Za-z0-9_-]{1,64})/breed", path)
                if m:
                    run = get_run(m.group(1))
                    st = run.state["settings"]
                    gen = breed_generation(
                        run,
                        body.get("selected") or [],
                        body.get("n_offspring", st["n_offspring"]),
                        bool(body.get("elitism", st["elitism"])))
                    self._send_json(HTTPStatus.OK,
                                    generation_payload(run, gen["index"]))
                    return
                m = re.fullmatch(r"/api/runs/([A-Za-z0-9_-]{1,64})/restart", path)
                if m:
                    run = get_run(m.group(1))
                    restart_run(run)
                    self._send_json(HTTPStatus.OK, generation_payload(run, 0))
                    return
                self._not_found()
        except ApiError as e:
            self._send_json(e.status, {"error": e.message})


def main() -> None:
    global DEBUG, RUNS_DIR, SAMPLES_DIR, VAR_KEEP, PORT
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--debug", action="store_true", help="log every request")
    parser.add_argument("--runs-dir", default=str(RUNS_DIR),
                        help="where new runs are created and listed from")
    parser.add_argument("--samples", default=str(SAMPLES_DIR),
                        help="seed SVG directory (used when creating a run)")
    parser.add_argument("--var-keep", type=float, default=VAR_KEEP,
                        help="fraction of variance to keep when fitting the "
                             "PCA basis for a new run (1.0 = full rank)")
    args = parser.parse_args()
    DEBUG = args.debug
    RUNS_DIR = Path(args.runs_dir).expanduser().resolve()
    SAMPLES_DIR = Path(args.samples).expanduser().resolve()
    VAR_KEEP = args.var_keep
    PORT = args.port

    n_seeds = len(list(SAMPLES_DIR.glob("vector_*.svg")))
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    mode = " [debug]" if DEBUG else ""
    print(f"serving on http://{HOST}:{PORT} — {n_seeds} seed SVGs, "
          f"runs in {RUNS_DIR}{mode}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.server_close()


if __name__ == "__main__":
    main()
