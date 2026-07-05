"""Visualize what ``preference_server.py`` learned about the eigenshape space.

``eigen_display.py`` shows what each PCA axis *does*; this shows which values
along those axes you *prefer* -- and, since a model is fitted **per size
bucket**, how that preference looks at each size. It reads a session
(``pref_data/session.json`` for the pinned basis; any stored ``n_active`` is
ignored -- every model built here learns/varies **all** components, matching
``preference_model.py``'s current philosophy) and its logged duels
(``pref_data/votes.jsonl``), refits one ``PreferenceModel`` per size bucket,
and writes a self-contained, **interactive** ``preference_results.html`` (like
``sample_display.py``: one embedded JSON blob per page, no server, JS reused
in spirit -- ``escapeHtml``, ``niceTicks``, ``scaleLinear``) with one section
per size (default: every bucket with at least one vote, in session order;
narrow it with ``--size``). Each section has, top to bottom:

* a **header pair** -- the population mean next to the model's current best
  guess (``PreferenceModel.best_coeffs()``), with that bucket's vote/tie count
  and a breakdown by duel mode (axis/blend/confirm);
* a **Bezier delta view** -- the best mark in ink over the mean in grey (no
  arrows -- the overlay itself carries the shape difference);
* an **evolution filmstrip** -- the model refit on prefixes of that bucket's
  vote log (checkpoints every ``max(5, n//12)`` votes, incrementally --
  ``observe()`` warm-starts IRLS from the last fit, so this doesn't refit from
  scratch), ``best_coeffs()`` decoded at each checkpoint;
* **per-axis z\\* trajectories** -- small-multiple line charts of each axis's
  preferred z\\* against vote count (shaded by its posterior ``zstar_std``, a
  zero line marking the mean, gaps where the axis had no interior peak yet),
  for the top ~8 axes by final utility span if there are many;
* **per-axis utility** -- both views side by side, since the overlay alone
  didn't show enough on its own: an **overlay** (every step of the walk
  superimposed in a single SVG, each step's stroke colored on a sequential
  green ramp -- pale = low learned utility, deep = high, the report's
  existing green="best" hue family -- drawn low-utility-first so the
  preferred shapes sit on top, the z\\* step drawn slightly thicker) beside
  the older **stepped-cells table** (one mark per step, cell background
  tinted white->green by that step's utility, a utility bar under each mark,
  the z\\*-nearest cell outlined) and a **profile** cell (the mark at
  z\\*). Labeled with the axis's posterior spread (``z* = 0.8 +/- 0.3``)
  and a settled/unsettled tag; and
* an **eigenspace scatter** -- the seed marks, the population mean, and the
  learned preference (``best_coeffs()``) plotted on two chosen components
  (selects labeled ``PC# (x.x% var)``), with an off-by-default checkbox to
  also plot every duel candidate this bucket has seen, colored by duel mode,
  filled for the winner and hollow for the loser (both hollow on a tie).
  Hovering or clicking a duel candidate highlights its opponent -- a dashed
  link line between the two, both points enlarged, every other duel point
  faded to low opacity -- so you can tell which candidates actually dueled
  each other.

Also, per size section:

* **per-axis utility curves** -- the MAP utility curve (``U(z) = w_lin*z +
  w_quad*z^2``) for the same top axes as the utility table, ~20 thin
  posterior draws (``model.sample_w``) underneath showing how sure the model
  is, a dashed line at z\\* when there's an interior peak, and evidence ticks
  along the bottom marking every "axis"-mode duel that actually probed that
  axis (filled ink = winner, hollow grey = loser, half-height grey = tie);
* a **model calibration** section -- a 5-bin reliability chart of predicted
  vs. observed outcome (sequential predictions made *before* each vote was
  folded in, with the Brier score) plus an **upset gallery** of the votes the
  model was most confident about and still got wrong;
* a **"when to stop voting"** panel -- total posterior z\\* uncertainty
  (``zstar_std`` std-weighted and summed across axes -- exactly the
  scheduler's axis-picking score) against vote count; flattening is the
  stopping signal;
* a **small-size legibility strip** near the header pair -- the current best
  guess rendered at 16/24/32/48/64px; and
* a **nearest-seed proximity** view after the Bezier delta -- the 3 seeds
  standardized-closest to the learned best guess, plus the median seed-seed
  distance for scale.

Once >=2 size buckets have votes, a document-level **cross-size comparison**
follows every section: each bucket's best guess overlaid in one SVG (one
categorical color per size) and a z\\* dot-plot per axis across sizes (error
bars = zstar_std). With fewer than 2 buckets voted on, a one-line note renders
in its place -- which is what the bundled example data (single "60px" bucket)
produces.

Refits from the log on each run, like ``eigen_display.py`` refits from
``Samples/``. Pure standard library; output is a generated artifact.

    python3 preference_display.py                 # every size with votes
    python3 preference_display.py --size 40px,60px
    python3 preference_display.py -n 8 --steps 11
    python3 preference_display.py --data-dir DIR -o out.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

from eigen import PCABasis
from genome import PATH_ORDER, STROKE_WIDTH, VIEWBOX
from preference_model import WINNER_Y, PreferenceModel, _sigmoid

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "pref_data"

# Mirrors preference_server.SIZES/DEFAULT_SIZE -- used only as a fallback
# ordering/label for a session predating the "sizes" key entirely.
DEFAULT_SIZE_ORDER = ["20px", "30px", "40px", "50px", "60px", "70px", "80px"]
DEFAULT_SIZE = "40px"
MODE_ORDER = ("axis", "blend", "confirm")

MEAN_STROKE = "#bbbbbb"
INK_STROKE = "#231f20"
TRAJ_COLOR = "#2a78d6"
SETTLED_ZSTD_STD = 0.35      # zstar_std under this (+ an interior peak) => "settled"

# Sequential utility ramp: a single hue (green, this report's existing
# "best"/peak color family -- #1a7f37 sits between these two steps) stepped
# light->dark per the dataviz skill's sequential-color rule, rather than
# introducing a second hue. Low utility is pale, not white, so a stroke on a
# white mark background stays visible at the low end.
UTIL_RAMP_LO = (0xcf, 0xe9, 0xd8)   # pale sage (reused as the old profile-cell border)
UTIL_RAMP_HI = (0x0f, 0x5c, 0x27)   # deep green

# Fixed categorical assignment for duel mode, in the eigenspace scatter --
# drawn from the dataviz skill's validated palette, skipping its green slot
# (slot 4, #008300) since this page already reserves green for "the learned
# preference" and a mode-colored dot shouldn't visually compete with that.
MODE_COLORS = {
    "axis": "#2a78d6",       # blue
    "blend": "#1baf7a",      # aqua
    "confirm": "#eda100",    # yellow
    "unspecified": "#4a3aa7",  # violet
}
BEST_COLOR = "#1a7f37"
SEED_COLOR = "#9a9a9a"
MEAN_MARKER_COLOR = "#666666"

# Categorical palette for the cross-size comparison (dataviz skill's validated
# 8-hue theme, fixed order -- unlike MODE_COLORS above, green isn't reserved
# here: every size *is* "the learned preference" for that bucket, so there's
# no single "preference" hue to protect.
SIZE_PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
               "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]

# VIEWBOX ("0 0 W H") aspect ratio, for the small-size legibility strip's
# fixed-px cells (mirrors eigen_explorer.py's AR).
_VB_W, _VB_H = (float(x) for x in VIEWBOX.split()[2:])
MARK_ASPECT = _VB_H / _VB_W


# ---------------------------------------------------------------------------
# Load a session + its votes (mirrors preference_server's persistence)
# ---------------------------------------------------------------------------

def load_session(data_dir: Path):
    session_path = data_dir / "session.json"
    if not session_path.is_file():
        raise SystemExit(f"no session.json in {data_dir} -- run preference_server.py first")
    state = json.loads(session_path.read_text(encoding="utf-8"))
    basis = PCABasis.from_dict(state["basis"])
    votes = []
    votes_path = data_dir / "votes.jsonl"
    if votes_path.is_file():
        for line in votes_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (rec.get("session") == state["session_id"]
                    and rec.get("winner") in WINNER_Y
                    and isinstance(rec.get("a_coeffs"), list)
                    and isinstance(rec.get("b_coeffs"), list)):
                votes.append(rec)
    votes.sort(key=lambda v: v.get("ts", ""))   # already log order; defensive
    return state, basis, votes


def group_by_size(votes: list[dict], default_size: str) -> dict[str, list[dict]]:
    by_size: dict[str, list[dict]] = {}
    for v in votes:
        by_size.setdefault(v.get("size") or default_size, []).append(v)
    return by_size


def size_order(state: dict, by_size: dict) -> list[str]:
    order = list(state.get("sizes") or DEFAULT_SIZE_ORDER)
    for s in by_size:
        if s not in order:
            order.append(s)
    return order


# ---------------------------------------------------------------------------
# Small chart helpers (plain-python SVG, no client-side JS -- see CLAUDE.md's
# note that these display scripts are static artifacts)
# ---------------------------------------------------------------------------

def _lin_scale(d0: float, d1: float, r0: float, r1: float):
    if d1 == d0:
        return lambda v: r0
    m = (r1 - r0) / (d1 - d0)
    return lambda v: r0 + (v - d0) * m


def _round(seq, nd: int = 6) -> list[float]:
    return [round(float(v), nd) for v in seq]


def _percentile(sorted_vals: list[float], q: float) -> float:
    """``q`` in [0,1] on an already-sorted list -- nearest-rank, no interpolation
    (fine at the draw counts used here); used to clip a wild posterior draw's
    influence on a chart's Y range without discarding the draw itself."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    idx = max(0, min(n - 1, round(q * (n - 1))))
    return sorted_vals[idx]


def _safe_id(s: str) -> str:
    """Size labels (``"40px"``) are already valid HTML id characters, but
    sanitize anyway so a stray size string can't break element ids."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)


def _nice_ticks(lo: float, hi: float, count: int) -> list[float]:
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    span = hi - lo
    step0 = span / count
    mag = 10 ** math.floor(math.log10(step0)) if step0 > 0 else 1.0
    norm = step0 / mag
    if norm < 1.5:
        step = mag
    elif norm < 3:
        step = 2 * mag
    elif norm < 7:
        step = 5 * mag
    else:
        step = 10 * mag
    start = math.ceil(lo / step) * step
    ticks = []
    v = start
    while v <= hi + 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


# ---------------------------------------------------------------------------
# Mark rendering
# ---------------------------------------------------------------------------

def _paths_at_coeffs(basis: PCABasis, coeffs) -> list[str]:
    g = basis.decode(coeffs)
    return [g.paths[pid].to_d() for pid in PATH_ORDER]


def _svg(path_ds: list[str], width: float, cls: str = "mark") -> str:
    body = "".join(
        f'<path d="{d}" fill="none" stroke="#231f20" '
        f'stroke-width="{width}" stroke-miterlimit="10"/>'
        for d in path_ds
    )
    return (f'<svg viewBox="{VIEWBOX}" preserveAspectRatio="xMidYMid meet" '
            f'class="{cls}">{body}</svg>')


def _util_ramp_color(norm: float) -> str:
    """norm in [0,1] -> pale sage (low learned utility) to deep green (high),
    a single-hue sequential ramp per the dataviz skill (never a rainbow).
    Used for the per-axis overlay's stroke color."""
    lo, hi = UTIL_RAMP_LO, UTIL_RAMP_HI
    r = round(lo[0] + (hi[0] - lo[0]) * norm)
    g = round(lo[1] + (hi[1] - lo[1]) * norm)
    b = round(lo[2] + (hi[2] - lo[2]) * norm)
    return "#%02x%02x%02x" % (r, g, b)


def _util_bg(norm: float) -> str:
    """norm in [0,1] -> white (low utility) to green (high utility). Used for
    the stepped-cells table's cell background -- a second, older encoding of
    the same learned-utility quantity as ``_util_ramp_color``, restored
    alongside the overlay because the overlay alone didn't read clearly."""
    r = round((1 - norm) * 255 + norm * 0x1a)
    g = round((1 - norm) * 255 + norm * 0x7f)
    b = round((1 - norm) * 255 + norm * 0x37)
    return "#%02x%02x%02x" % (r, g, b)


LEGIB_SIZES = (16, 24, 32, 48, 64)


def legibility_strip_html(basis: PCABasis, model: PreferenceModel) -> str:
    """The current best guess rendered at fixed small pixel heights (like
    eigen_explorer.py's preview strip) -- a quick check that the learned mark
    still reads once it's actually small."""
    ds = _paths_at_coeffs(basis, model.best_coeffs())
    svg = _svg(ds, STROKE_WIDTH, "legib-svg")
    cells = []
    for h in LEGIB_SIZES:
        w = round(h / MARK_ASPECT, 1)
        cells.append(
            f'<div class="legib-cell"><div class="legib-mark" '
            f'style="width:{w}px;height:{h}px">{svg}</div>'
            f'<div class="legib-cap">{h}px</div></div>')
    return (f'<div class="legib-row">{"".join(cells)}</div>'
           f'<p class="chart-sub">Legibility check at small sizes.</p>')


def _zdist(c1: list[float], c2: list[float], stds: list[float]) -> float:
    """Standardized z-distance between two full-K coefficient vectors --
    mirrors ``PreferenceModel._zdist``."""
    return math.sqrt(sum(((a - b) / s) ** 2 for a, b, s in zip(c1, c2, stds)))


def seed_proximity_html(basis: PCABasis, model: PreferenceModel, state: dict) -> str:
    """The 3 seeds standardized-closest to the learned best guess, with the
    median seed<->seed distance as the yardstick for "is this close to an
    existing design, or somewhere new"."""
    seeds = build_seed_points(basis, state)
    if len(seeds) < 2:
        return '<p class="chart-sub">Not enough seeds recorded to judge proximity.</p>'

    stds = basis.stds
    best = model.best_coeffs()
    dists = sorted(
        ((_zdist(best, s["coeffs"], stds), s["name"], s["coeffs"]) for s in seeds),
        key=lambda t: t[0])
    nearest = dists[:3]

    pair_ds = sorted(
        _zdist(seeds[i]["coeffs"], seeds[j]["coeffs"], stds)
        for i in range(len(seeds)) for j in range(i + 1, len(seeds)))
    m = len(pair_ds)
    median = pair_ds[m // 2] if m % 2 else (pair_ds[m // 2 - 1] + pair_ds[m // 2]) / 2.0

    best_g = basis.decode(best)
    best_ds = [best_g.paths[pid].to_d() for pid in PATH_ORDER]

    items = []
    for dist, name, scoeffs in nearest:
        seed_g = basis.decode(scoeffs)
        seed_ds = [seed_g.paths[pid].to_d() for pid in PATH_ORDER]
        body = (
            "".join(f'<path d="{d}" fill="none" stroke="{MEAN_STROKE}" '
                   f'stroke-width="{STROKE_WIDTH}" stroke-miterlimit="10"/>' for d in seed_ds)
            + "".join(f'<path d="{d}" fill="none" stroke="{INK_STROKE}" '
                     f'stroke-width="{STROKE_WIDTH}" stroke-miterlimit="10"/>' for d in best_ds)
        )
        svg = (f'<svg viewBox="{VIEWBOX}" preserveAspectRatio="xMidYMid meet" '
              f'class="mark mid">{body}</svg>')
        name_esc = html.escape(name) if name else "(unnamed seed)"
        items.append(
            f'<div class="seedprox-item">{svg}'
            f'<div class="seedprox-label">{name_esc}<br>dist {dist:.2f}</div></div>')

    nearest_dist = nearest[0][0]
    if nearest_dist <= median:
        verdict = ("closer to that seed than seeds typically sit from each other -- "
                  "reads as close to an existing design")
    else:
        verdict = ("farther from every seed than seeds typically sit from each other -- "
                  "reads as a genuinely new point in the design space")
    summary = (f'<p class="chart-sub">Median seed&harr;seed distance: {median:.2f}. Nearest '
              f'seed to the learned best is <b>{html.escape(nearest[0][1]) or "(unnamed)"}</b> '
              f'at {nearest_dist:.2f} &mdash; {verdict}.</p>')
    return f'<div class="seedprox-row">{"".join(items)}</div>{summary}'


def _delta_view_svg(mean_g, best_g) -> str:
    """Best mark (ink) over the population mean (grey) -- no arrows; the
    overlay itself is the delta."""
    mean_ds = [mean_g.paths[pid].to_d() for pid in PATH_ORDER]
    best_ds = [best_g.paths[pid].to_d() for pid in PATH_ORDER]
    body = (
        "".join(f'<path d="{d}" fill="none" stroke="{MEAN_STROKE}" '
               f'stroke-width="{STROKE_WIDTH}" stroke-miterlimit="10"/>' for d in mean_ds)
        + "".join(f'<path d="{d}" fill="none" stroke="{INK_STROKE}" '
                 f'stroke-width="{STROKE_WIDTH}" stroke-miterlimit="10"/>' for d in best_ds)
    )
    return (f'<svg viewBox="{VIEWBOX}" preserveAspectRatio="xMidYMid meet" '
          f'class="mark big">{body}</svg>')


# ---------------------------------------------------------------------------
# Evolution filmstrip: one model, observed incrementally, snapshotted
# ---------------------------------------------------------------------------

def build_frames_and_model(basis: PCABasis, votes: list[dict], seed: int = 0):
    """Refit on checkpoints of ``votes`` without ever refitting from scratch:
    one ``PreferenceModel``, ``observe()``-d one vote at a time (IRLS warm-
    starts from the previous fit), snapshotting at each checkpoint. The model
    returned is therefore also the final, fully-observed model for this size
    bucket -- callers reuse it for the header/delta/utility-rows views instead
    of building a second one.

    Always built with ``n_active=None`` (all components) -- matching
    ``preference_model.py``'s current philosophy that every axis is learned
    and varied, just scheduled unevenly. Any ``n_active`` stored in an old
    session is ignored here.

    Also returns ``preds``: one dict per vote, the model's *sequential*
    prediction made just before that vote was folded in --
    ``p = sigmoid(utility(a) - utility(b))`` against ``y = WINNER_Y[winner]``
    -- used by the calibration view and the upset gallery. ``n_obs_before``
    records how many votes the model had already seen at that point, since
    "how surprising was this" is only meaningful once the model has some
    data."""
    model = PreferenceModel(basis.stds, n_active=None, rng=random.Random(seed))
    n = len(votes)
    step = max(5, n // 12) if n else 5
    checkpoints = sorted(set(list(range(step, n, step)) + ([n] if n else [])))

    frames = []

    def snapshot(count):
        frames.append({
            "n": count,
            "coeffs": model.best_coeffs(),
            "z": model.preferred_z(),        # [(z, is_peak)] len M
            "zstd": model.zstar_stds(),       # [float] len M
        })

    snapshot(0)
    preds = []
    ci = 0
    for i, v in enumerate(votes, start=1):
        p = _sigmoid(model.utility(v["a_coeffs"]) - model.utility(v["b_coeffs"]))
        preds.append({
            "idx": i,
            "p": p,
            "y": WINNER_Y[v["winner"]],
            "winner": v["winner"],
            "mode": v.get("mode") or "unspecified",
            "n_obs_before": model.n_obs,
            "a_coeffs": v["a_coeffs"],
            "b_coeffs": v["b_coeffs"],
        })
        model.observe(v["a_coeffs"], v["b_coeffs"], v["winner"])
        if ci < len(checkpoints) and i == checkpoints[ci]:
            snapshot(i)
            ci += 1
    return frames, model, preds


def filmstrip_html(basis: PCABasis, frames: list[dict]) -> str:
    cells = []
    for f in frames:
        ds = _paths_at_coeffs(basis, f["coeffs"])
        label = "prior" if f["n"] == 0 else f'{f["n"]} votes'
        cells.append(
            f'<div class="film-cell">{_svg(ds, STROKE_WIDTH, "mark film-mark")}'
            f'<div class="film-label">{label}</div></div>'
        )
    return f'<div class="film-row">{"".join(cells)}</div>'


# ---------------------------------------------------------------------------
# Per-axis z* trajectories
# ---------------------------------------------------------------------------

TRAJ_W, TRAJ_H = 220, 130
TRAJ_M = {"left": 32, "right": 10, "top": 10, "bottom": 22}


def traj_chart_html(k: int, entry: dict, frames: list[dict], basis: PCABasis,
                    total_var: float) -> str:
    n_total = frames[-1]["n"] if frames else 0
    var_pct = (basis.eigenvalues[k] / total_var * 100.0) if total_var else 0.0
    title = f'<div class="traj-title">PC{k + 1}</div><div class="traj-sub">{var_pct:.1f}% var</div>'

    flags = [f["z"][k][1] for f in frames]
    runs, cur = [], []
    for i, isp in enumerate(flags):
        if isp:
            cur.append(i)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)

    if not runs:
        return (f'<div class="traj-item">{title}'
               f'<div class="traj-empty">no interior peak yet<br>(edge preference)</div></div>')

    ys = [0.0]
    for run in runs:
        for i in run:
            z, zstd = frames[i]["z"][k][0], frames[i]["zstd"][k]
            ys += [z - zstd, z + zstd]
    y0, y1 = min(ys), max(ys)
    pad = (y1 - y0) * 0.15 or 0.5
    y0, y1 = y0 - pad, y1 + pad
    x1 = max(n_total, 1)

    sx = _lin_scale(0, x1, TRAJ_M["left"], TRAJ_W - TRAJ_M["right"])
    sy = _lin_scale(y0, y1, TRAJ_H - TRAJ_M["bottom"], TRAJ_M["top"])

    svg = []
    for t in _nice_ticks(y0, y1, 3):
        py = sy(t)
        svg.append(f'<line class="grid" x1="{TRAJ_M["left"]}" x2="{TRAJ_W - TRAJ_M["right"]}" '
                   f'y1="{py:.1f}" y2="{py:.1f}"/>')
        svg.append(f'<text x="{TRAJ_M["left"] - 5}" y="{py + 3:.1f}" text-anchor="end">{t:g}</text>')
    zy = sy(0.0)
    svg.append(f'<line class="zero-line" x1="{TRAJ_M["left"]}" x2="{TRAJ_W - TRAJ_M["right"]}" '
               f'y1="{zy:.1f}" y2="{zy:.1f}"/>')
    svg.append(f'<line class="baseline" x1="{TRAJ_M["left"]}" x2="{TRAJ_W - TRAJ_M["right"]}" '
               f'y1="{TRAJ_H - TRAJ_M["bottom"]}" y2="{TRAJ_H - TRAJ_M["bottom"]}"/>')
    svg.append(f'<line class="baseline" x1="{TRAJ_M["left"]}" x2="{TRAJ_M["left"]}" '
               f'y1="{TRAJ_M["top"]}" y2="{TRAJ_H - TRAJ_M["bottom"]}"/>')
    svg.append(f'<text x="{TRAJ_M["left"]}" y="{TRAJ_H - 4}" text-anchor="start">0</text>')
    svg.append(f'<text x="{TRAJ_W - TRAJ_M["right"]}" y="{TRAJ_H - 4}" text-anchor="end">{n_total}</text>')

    for run in runs:
        if len(run) == 1:
            i = run[0]
            n, z, zstd = frames[i]["n"], frames[i]["z"][k][0], frames[i]["zstd"][k]
            px = sx(n)
            svg.append(f'<line class="traj-err" x1="{px:.1f}" x2="{px:.1f}" '
                       f'y1="{sy(z - zstd):.1f}" y2="{sy(z + zstd):.1f}"/>')
            svg.append(f'<circle class="traj-dot" cx="{px:.1f}" cy="{sy(z):.1f}" r="3"/>')
            continue
        top = [(sx(frames[i]["n"]), sy(frames[i]["z"][k][0] + frames[i]["zstd"][k])) for i in run]
        bot = [(sx(frames[i]["n"]), sy(frames[i]["z"][k][0] - frames[i]["zstd"][k])) for i in run]
        band_pts = top + list(reversed(bot))
        band_d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in band_pts) + " Z"
        line_d = "M" + " L".join(
            f"{sx(frames[i]['n']):.1f},{sy(frames[i]['z'][k][0]):.1f}" for i in run)
        svg.append(f'<path class="traj-band" d="{band_d}"/>')
        svg.append(f'<path class="traj-line" d="{line_d}"/>')
        last_i = run[-1]
        svg.append(f'<circle class="traj-dot" cx="{sx(frames[last_i]["n"]):.1f}" '
                   f'cy="{sy(frames[last_i]["z"][k][0]):.1f}" r="3"/>')

    if entry["peak"] and runs[-1][-1] == len(frames) - 1:
        last_i = runs[-1][-1]
        lx, ly = sx(frames[last_i]["n"]), sy(frames[last_i]["z"][k][0])
        svg.append(f'<text class="traj-final" x="{lx + 5:.1f}" y="{ly + 3:.1f}">'
                   f'{entry["z_star"]:+.2f}</text>')

    body = (f'<svg width="{TRAJ_W}" height="{TRAJ_H}" viewBox="0 0 {TRAJ_W} {TRAJ_H}" '
           f'class="axis traj-svg">{"".join(svg)}</svg>')
    return f'<div class="traj-item">{title}{body}</div>'


def trajectories_html(model: PreferenceModel, frames: list[dict], basis: PCABasis) -> str:
    total_var = sum(basis.eigenvalues) or 1.0
    report = model.axis_report()                # sorted by span, descending
    shown = report[:min(8, len(report))]
    items = "".join(traj_chart_html(r["axis"], r, frames, basis, total_var) for r in shown)
    return f'<div class="traj-grid">{items}</div>'


# ---------------------------------------------------------------------------
# "When to stop voting": total posterior z* uncertainty (std-weighted, summed
# across axes -- exactly the "axis" scheduler's per-axis score) vs. vote
# count. Flattening is a stopping signal: further duels are pinning down
# less and less.
# ---------------------------------------------------------------------------

STOP_W, STOP_H = 460, 150
STOP_M = {"left": 46, "right": 14, "top": 12, "bottom": 26}


def stopping_panel_html(frames: list[dict], basis: PCABasis) -> str:
    pts = [(f["n"], sum(zstd * std for zstd, std in zip(f["zstd"], basis.stds)))
          for f in frames]
    caption = ('<p class="chart-sub">Sum of every axis\'s posterior z&#42; uncertainty '
              '(&times; that axis\'s std -- exactly the &quot;axis&quot; scheduler\'s '
              'per-axis score), against vote count. When this flattens, additional duels '
              'are pinning down less and less -- a stopping signal, not a hard rule.</p>')
    if len(pts) < 2:
        return '<p class="chart-sub">Not enough votes yet to plot a trend.</p>' + caption

    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1 = 0, max(xs) or 1
    y0 = 0.0
    y1 = max(ys) * 1.1 or 1.0

    sx = _lin_scale(x0, x1, STOP_M["left"], STOP_W - STOP_M["right"])
    sy = _lin_scale(y0, y1, STOP_H - STOP_M["bottom"], STOP_M["top"])

    svg = []
    for t in _nice_ticks(y0, y1, 4):
        py = sy(t)
        svg.append(f'<line class="grid" x1="{STOP_M["left"]}" x2="{STOP_W - STOP_M["right"]}" '
                   f'y1="{py:.1f}" y2="{py:.1f}"/>')
        svg.append(f'<text x="{STOP_M["left"] - 6}" y="{py + 3:.1f}" text-anchor="end">{t:g}</text>')
    for t in _nice_ticks(x0, x1, 5):
        px = sx(t)
        svg.append(f'<text x="{px:.1f}" y="{STOP_H - STOP_M["bottom"] + 14}" '
                   f'text-anchor="middle">{t:g}</text>')
    svg.append(f'<line class="baseline" x1="{STOP_M["left"]}" x2="{STOP_W - STOP_M["right"]}" '
               f'y1="{STOP_H - STOP_M["bottom"]}" y2="{STOP_H - STOP_M["bottom"]}"/>')
    svg.append(f'<line class="baseline" x1="{STOP_M["left"]}" x2="{STOP_M["left"]}" '
               f'y1="{STOP_M["top"]}" y2="{STOP_H - STOP_M["bottom"]}"/>')

    line_d = "M" + " L".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
    svg.append(f'<path class="stop-line" d="{line_d}"/>')
    for x, y in pts:
        svg.append(f'<circle class="stop-dot" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3"/>')

    body = (f'<svg width="{STOP_W}" height="{STOP_H}" viewBox="0 0 {STOP_W} {STOP_H}" '
           f'class="axis stop-svg">{"".join(svg)}</svg>')
    return body + caption


# ---------------------------------------------------------------------------
# Header pair
# ---------------------------------------------------------------------------

def header_pair_html(basis: PCABasis, model: PreferenceModel, votes: list[dict]) -> str:
    mean_ds = _paths_at_coeffs(basis, [0.0] * basis.n_components)
    best_ds = _paths_at_coeffs(basis, model.best_coeffs())
    n_ties = sum(1 for v in votes if v["winner"] == "tie")
    counts = Counter(v.get("mode") or "unspecified" for v in votes)
    order = [m for m in MODE_ORDER if m in counts] + sorted(
        m for m in counts if m not in MODE_ORDER)
    mode_str = " &middot; ".join(f"{m} {counts[m]}" for m in order)
    note = ("" if model.n_obs >= 15 else
           f'<p class="warn">Only {model.n_obs} vote(s) so far for this size -- '
           f'the model is still close to its prior.</p>')
    return f"""
  <div class="summary">
    <figure>{_svg(mean_ds, STROKE_WIDTH, "big")}<figcaption>population mean</figcaption></figure>
    <div class="arrow">&rarr;</div>
    <figure>{_svg(best_ds, STROKE_WIDTH, "big")}<figcaption>current best guess</figcaption></figure>
    <div class="summary-stats">
      <div class="stat-n"><b>{model.n_obs}</b> duels ({n_ties} ties)</div>
      <div class="stat-modes">{mode_str or "no mode recorded"}</div>
    </div>
  </div>
  {note}"""


# ---------------------------------------------------------------------------
# Per-axis utility overlay: the walk across one axis superimposed in a
# single SVG, stroke-colored by learned utility -- one figure per axis
# instead of one cell per step. Rendered alongside (not instead of) the
# stepped-cells table below, since the overlay alone didn't show enough on
# its own.
# ---------------------------------------------------------------------------

def _axis_overlay_svg(basis: PCABasis, k: int, std: float, lin: float, quad: float,
                      z_star: float, zs: list[float]) -> str:
    """Every step of the walk across axis ``k``, superimposed: stroke color
    encodes that step's learned utility on the sequential green ramp (pale =
    low, deep = high), and steps are drawn low-utility-first so the preferred
    shapes end up on top. The step nearest ``z_star`` is stroked thicker."""
    us = [lin * z + quad * z * z for z in zs]
    u_star = lin * z_star + quad * z_star * z_star
    lo = min(us + [u_star])
    span = (max(us + [u_star]) - lo) or 1.0
    star_i = min(range(len(zs)), key=lambda i: abs(zs[i] - z_star))

    order = sorted(range(len(zs)), key=lambda i: us[i])   # low utility first
    layers = []
    for i in order:
        z = zs[i]
        norm = (us[i] - lo) / span
        ds = _paths_at_coeffs(
            basis, [z * std if j == k else 0.0 for j in range(basis.n_components)])
        color = _util_ramp_color(norm)
        width = STROKE_WIDTH * (2.0 if i == star_i else 1.0)
        layers.append("".join(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width:.2f}" '
            f'stroke-miterlimit="10"/>' for d in ds))
    body = "".join(layers)
    return (f'<svg viewBox="{VIEWBOX}" preserveAspectRatio="xMidYMid meet" '
          f'class="mark util-overlay-svg">{body}</svg>')


def utility_rows_html(basis: PCABasis, model: PreferenceModel, n_components: int,
                      z_max: float, steps: int) -> str:
    """One table: row label, the whole-walk overlay, the restored stepped
    cells (tinted white->green by ``_util_bg``, a utility bar, the z*-nearest
    cell outlined), then a profile cell -- the overlay and the stepped cells
    are two encodings of the same learned-utility numbers, kept side by side
    because the overlay on its own didn't show enough."""
    if steps < 3:
        steps = 3
    if steps % 2 == 0:
        steps += 1
    half = steps // 2
    zs = [(-z_max + 2 * z_max * k / (steps - 1)) for k in range(steps)]
    zs[half] = 0.0

    report = model.axis_report()
    n_show = min(n_components, len(report))

    col_head = "".join(
        f'<th>{("mean" if abs(z) < 1e-9 else f"{z:+.2g}σ")}</th>' for z in zs
    )

    rows = []
    for r in report[:n_show]:
        k = r["axis"]
        std = basis.stds[k]
        lin, quad = r["lin"], r["quad"]
        z_star = r["z_star"]
        overlay = _axis_overlay_svg(basis, k, std, lin, quad, z_star, zs)

        us = [lin * z + quad * z * z for z in zs]
        u_star = lin * z_star + quad * z_star * z_star
        lo = min(us + [u_star])
        span = (max(us + [u_star]) - lo) or 1.0
        star_k = min(range(steps), key=lambda i: abs(zs[i] - z_star))

        cells = []
        for i, z in enumerate(zs):
            norm = (us[i] - lo) / span
            ds = _paths_at_coeffs(
                basis, [z * std if j == k else 0.0 for j in range(basis.n_components)])
            is_star = i == star_k
            cls = "cell star" if is_star else "cell"
            bar_h = round(2 + 22 * norm)
            cells.append(
                f'<td class="{cls}" style="background:{_util_bg(norm)}">'
                f'{_svg(ds, STROKE_WIDTH)}'
                f'<div class="bar"><i style="height:{bar_h}px"></i></div></td>')

        star_ds = _paths_at_coeffs(
            basis, [z_star * std if j == k else 0.0 for j in range(basis.n_components)])
        kind = "peak" if r["peak"] else "edge"
        profile_cell = (
            f'<td class="cell profile">{_svg(star_ds, STROKE_WIDTH)}'
            f'<div class="ptag">{kind} @ {z_star:+.2f}&sigma;</div></td>')

        settled = r["peak"] and r["zstar_std"] < SETTLED_ZSTD_STD
        settle_cls = "settled" if settled else "unsettled"
        label = (f'<div class="pc">PC{k + 1}</div>'
                 f'<div class="star">z* = {z_star:+.2f} &plusmn; {r["zstar_std"]:.2f}</div>'
                 f'<div class="kind {kind}">{kind}</div>'
                 f'<div class="settle-tag {settle_cls}">{settle_cls}</div>')

        rows.append(
            f'<tr><th class="rowlabel"><div class="util-label">{label}</div></th>'
            f'<td class="cell overlay">{overlay}</td>'
            f'{"".join(cells)}{profile_cell}</tr>')

    legend = (f'<div class="ramp-legend"><span class="ramp-swatch" '
             f'style="background:linear-gradient(to right, {_util_ramp_color(0.0)}, '
             f'{_util_ramp_color(1.0)})"></span>'
             f'<span>low learned preference &rarr; high, along each axis\'s own walk '
             f'(not comparable across axes) -- encoded twice: the overlay\'s stroke '
             f'color and the stepped cells\' background tint.</span></div>')

    return f"""<table>
    <thead><tr><th></th><th>overlay</th>{col_head}<th>profile</th></tr></thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>{legend}"""


# ---------------------------------------------------------------------------
# Per-axis utility curves: what the model believes U(z) looks like (MAP,
# thick green), how sure it is (thin posterior draws, same green, low
# opacity), where it thinks the peak is (dashed line at z*), and the actual
# evidence (ticks along the bottom from "axis"-mode duels that probed this
# axis). Same axes/order as the utility table above (``model.axis_report()``,
# top n_show).
# ---------------------------------------------------------------------------

UCURVE_W, UCURVE_H = TRAJ_W, TRAJ_H
UCURVE_M = TRAJ_M
N_POSTERIOR_DRAWS = 20
UCURVE_SEED = 20260704   # fixed -> the posterior-draw fan is reproducible
UCURVE_TICK_H = 10


def _axis_evidence_ticks(k: int, votes: list[dict], basis: PCABasis) -> list[dict]:
    """Every "axis"-mode vote in this bucket that probed axis ``k``: uses the
    logged ``axis`` field when present, else infers it as the single index
    where a_coeffs/b_coeffs differ (true by construction for a staircase
    duel -- every other axis is held at the base point)."""
    out = []
    for i, v in enumerate(votes, start=1):
        if (v.get("mode") or "unspecified") != "axis":
            continue
        a, b = v["a_coeffs"], v["b_coeffs"]
        axis = v.get("axis")
        if axis is None:
            diffs = [j for j in range(len(a)) if abs(a[j] - b[j]) > 1e-9]
            if len(diffs) != 1:
                continue
            axis = diffs[0]
        if axis != k:
            continue
        std = basis.stds[k]
        out.append({"idx": i, "za": a[k] / std, "zb": b[k] / std, "winner": v["winner"]})
    return out


def utility_curve_chart_html(k: int, entry: dict, model: PreferenceModel,
                             votes: list[dict], basis: PCABasis, z_max: float) -> str:
    title = f'<div class="traj-title">PC{k + 1}</div>'
    lin, quad, z_star, is_peak = entry["lin"], entry["quad"], entry["z_star"], entry["peak"]

    n_grid = 61
    zs = [-z_max + 2 * z_max * i / (n_grid - 1) for i in range(n_grid)]
    map_ys = [lin * z + quad * z * z for z in zs]

    rng = random.Random(UCURVE_SEED + k)
    draw_curves = []
    all_draw_vals = []
    for _ in range(N_POSTERIOR_DRAWS):
        w = model.sample_w(rng)
        b, a = w[k], w[model.M + k]
        ys = [b * z + a * z * z for z in zs]
        draw_curves.append(ys)
        all_draw_vals.extend(ys)

    y0, y1 = min(map_ys), max(map_ys)
    if all_draw_vals:
        sv = sorted(all_draw_vals)
        y0 = min(y0, _percentile(sv, 0.05))
        y1 = max(y1, _percentile(sv, 0.95))
    if y1 - y0 < 1e-9:
        y0, y1 = y0 - 0.5, y1 + 0.5
    pad = (y1 - y0) * 0.12
    y0, y1 = y0 - pad, y1 + pad

    sx = _lin_scale(-z_max, z_max, UCURVE_M["left"], UCURVE_W - UCURVE_M["right"])
    sy = _lin_scale(y0, y1, UCURVE_H - UCURVE_M["bottom"], UCURVE_M["top"])
    clampy = lambda v: max(y0, min(y1, v))
    clampz = lambda v: max(-z_max, min(z_max, v))

    svg = []
    for t in _nice_ticks(y0, y1, 3):
        py = sy(t)
        svg.append(f'<line class="grid" x1="{UCURVE_M["left"]}" x2="{UCURVE_W - UCURVE_M["right"]}" '
                   f'y1="{py:.1f}" y2="{py:.1f}"/>')
        svg.append(f'<text x="{UCURVE_M["left"] - 5}" y="{py + 3:.1f}" text-anchor="end">{t:g}</text>')
    svg.append(f'<line class="baseline" x1="{UCURVE_M["left"]}" x2="{UCURVE_W - UCURVE_M["right"]}" '
               f'y1="{UCURVE_H - UCURVE_M["bottom"]}" y2="{UCURVE_H - UCURVE_M["bottom"]}"/>')
    svg.append(f'<line class="baseline" x1="{UCURVE_M["left"]}" x2="{UCURVE_M["left"]}" '
               f'y1="{UCURVE_M["top"]}" y2="{UCURVE_H - UCURVE_M["bottom"]}"/>')
    svg.append(f'<text x="{UCURVE_M["left"]}" y="{UCURVE_H - 4}" text-anchor="start">{-z_max:g}</text>')
    svg.append(f'<text x="{UCURVE_W - UCURVE_M["right"]}" y="{UCURVE_H - 4}" '
               f'text-anchor="end">{z_max:g}</text>')

    for ys in draw_curves:
        d = "M" + " L".join(f"{sx(z):.1f},{sy(clampy(y)):.1f}" for z, y in zip(zs, ys))
        svg.append(f'<path class="ucurve-draw" d="{d}"/>')

    if is_peak:
        px = sx(clampz(z_star))
        svg.append(f'<line class="ucurve-zstar" x1="{px:.1f}" x2="{px:.1f}" '
                   f'y1="{UCURVE_M["top"]}" y2="{UCURVE_H - UCURVE_M["bottom"]}"/>')

    map_d = "M" + " L".join(f"{sx(z):.1f},{sy(clampy(y)):.1f}" for z, y in zip(zs, map_ys))
    svg.append(f'<path class="ucurve-map" d="{map_d}"/>')

    ticky = UCURVE_H - UCURVE_M["bottom"]
    for ev in _axis_evidence_ticks(k, votes, basis):
        xa, xb = sx(clampz(ev["za"])), sx(clampz(ev["zb"]))
        title_a = f'vote #{ev["idx"]}: z={ev["za"]:+.2f}'
        title_b = f'vote #{ev["idx"]}: z={ev["zb"]:+.2f}'
        if ev["winner"] == "tie":
            h = UCURVE_TICK_H * 0.5
            svg.append(f'<line class="tick-tie" x1="{xa:.1f}" x2="{xa:.1f}" '
                       f'y1="{ticky}" y2="{ticky - h}"><title>{title_a}, tie</title></line>')
            svg.append(f'<line class="tick-tie" x1="{xb:.1f}" x2="{xb:.1f}" '
                       f'y1="{ticky}" y2="{ticky - h}"><title>{title_b}, tie</title></line>')
        else:
            win_x, lose_x = (xa, xb) if ev["winner"] == "a" else (xb, xa)
            win_t, lose_t = (title_a, title_b) if ev["winner"] == "a" else (title_b, title_a)
            svg.append(f'<line class="tick-win" x1="{win_x:.1f}" x2="{win_x:.1f}" '
                       f'y1="{ticky}" y2="{ticky - UCURVE_TICK_H}">'
                       f'<title>{win_t}, won</title></line>')
            svg.append(f'<line class="tick-lose" x1="{lose_x:.1f}" x2="{lose_x:.1f}" '
                       f'y1="{ticky}" y2="{ticky - UCURVE_TICK_H}">'
                       f'<title>{lose_t}, lost</title></line>')

    body = (f'<svg width="{UCURVE_W}" height="{UCURVE_H}" viewBox="0 0 {UCURVE_W} {UCURVE_H}" '
           f'class="axis ucurve-svg">{"".join(svg)}</svg>')
    return f'<div class="traj-item">{title}{body}</div>'


def utility_curves_html(model: PreferenceModel, votes: list[dict], basis: PCABasis,
                        z_max: float, n_components: int) -> str:
    report = model.axis_report()
    n_show = min(n_components, len(report))
    items = "".join(utility_curve_chart_html(r["axis"], r, model, votes, basis, z_max)
                    for r in report[:n_show])
    caption = ('<p class="chart-sub">The thick green curve is what the model currently '
              f'believes each axis\'s utility looks like; the pale thin curves are '
              f'{N_POSTERIOR_DRAWS} posterior draws (<code>sample_w</code>), showing how sure it '
              'is. The dashed vertical line marks z&#42; where there\'s an interior peak. '
              'Ticks along the bottom are the actual duel evidence for that axis: filled '
              'ink = winner, hollow grey = loser, half-height grey = tie.</p>')
    return f'<div class="traj-grid">{items}</div>{caption}'


# ---------------------------------------------------------------------------
# Model calibration: were the model's sequential predictions (made *before*
# each vote was folded in, by build_frames_and_model) any good? A reliability
# chart bins predicted vs. observed outcome; an upset gallery surfaces the
# votes the model was most confident about and still got wrong.
# ---------------------------------------------------------------------------

CAL_W, CAL_H = 260, 220
CAL_M = {"left": 40, "right": 14, "top": 14, "bottom": 34}
N_CAL_BINS = 5


def calibration_chart_html(preds: list[dict]) -> tuple[str, float]:
    """Returns (svg_html, brier_score). Bins predictions into ``N_CAL_BINS``
    equal-width bins over [0,1]; each bin is one dot (mean predicted vs. mean
    observed, radius scaled by bin count), plus the y=x reference diagonal."""
    bins: list[list[dict]] = [[] for _ in range(N_CAL_BINS)]
    for pr in preds:
        b = min(N_CAL_BINS - 1, int(pr["p"] * N_CAL_BINS))
        bins[b].append(pr)
    max_count = max((len(bb) for bb in bins), default=1) or 1

    brier = (sum((pr["p"] - pr["y"]) ** 2 for pr in preds) / len(preds)) if preds else 0.0

    sx = _lin_scale(0.0, 1.0, CAL_M["left"], CAL_W - CAL_M["right"])
    sy = _lin_scale(0.0, 1.0, CAL_H - CAL_M["bottom"], CAL_M["top"])

    svg = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        px, py = sx(t), sy(t)
        svg.append(f'<line class="grid" x1="{CAL_M["left"]}" x2="{CAL_W - CAL_M["right"]}" '
                   f'y1="{py:.1f}" y2="{py:.1f}"/>')
        svg.append(f'<line class="grid" x1="{px:.1f}" x2="{px:.1f}" '
                   f'y1="{CAL_M["top"]}" y2="{CAL_H - CAL_M["bottom"]}"/>')
        svg.append(f'<text x="{CAL_M["left"] - 6}" y="{py + 3:.1f}" text-anchor="end">{t:g}</text>')
        svg.append(f'<text x="{px:.1f}" y="{CAL_H - CAL_M["bottom"] + 14}" '
                   f'text-anchor="middle">{t:g}</text>')
    svg.append(f'<line class="baseline" x1="{CAL_M["left"]}" x2="{CAL_W - CAL_M["right"]}" '
               f'y1="{CAL_H - CAL_M["bottom"]}" y2="{CAL_H - CAL_M["bottom"]}"/>')
    svg.append(f'<line class="baseline" x1="{CAL_M["left"]}" x2="{CAL_M["left"]}" '
               f'y1="{CAL_M["top"]}" y2="{CAL_H - CAL_M["bottom"]}"/>')
    svg.append(f'<line class="cal-diag" x1="{sx(0):.1f}" y1="{sy(0):.1f}" '
               f'x2="{sx(1):.1f}" y2="{sy(1):.1f}"/>')

    for b in bins:
        if not b:
            continue
        mp = sum(pr["p"] for pr in b) / len(b)
        mo = sum(pr["y"] for pr in b) / len(b)
        r = round(min(12.0, max(3.0, 3 + 7 * math.log1p(len(b)) / math.log1p(max_count))), 1)
        px, py = sx(mp), sy(mo)
        svg.append(f'<circle class="cal-dot" cx="{px:.1f}" cy="{py:.1f}" r="{r}">'
                   f'<title>{len(b)} votes, mean predicted {mp:.2f}, mean observed '
                   f'{mo:.2f}</title></circle>')

    body = (f'<svg width="{CAL_W}" height="{CAL_H}" viewBox="0 0 {CAL_W} {CAL_H}" '
           f'class="axis cal-svg">{"".join(svg)}</svg>')
    return body, brier


def upset_gallery_html(basis: PCABasis, preds: list[dict], min_n_obs: int = 10,
                       top_k: int = 5) -> str:
    """Top ``top_k`` non-tie votes, restricted to predictions made once the
    model had already seen >= ``min_n_obs`` duels (so "surprising" means
    something), where the actual winner had the lowest predicted probability
    -- the model was most confident about the loser and still lost."""
    candidates = []
    for pr in preds:
        if pr["winner"] == "tie" or pr["n_obs_before"] < min_n_obs:
            continue
        p_winner = pr["p"] if pr["winner"] == "a" else (1.0 - pr["p"])
        candidates.append((p_winner, pr))
    candidates.sort(key=lambda t: t[0])
    top = candidates[:top_k]
    if not top:
        return (f'<p class="chart-sub">No upsets to show yet -- need non-tie votes made '
               f'once the model had already seen &ge;{min_n_obs} duels.</p>')

    items = []
    for p_winner, pr in top:
        winner_ds = _paths_at_coeffs(
            basis, pr["a_coeffs"] if pr["winner"] == "a" else pr["b_coeffs"])
        loser_ds = _paths_at_coeffs(
            basis, pr["b_coeffs"] if pr["winner"] == "a" else pr["a_coeffs"])
        win_svg = _svg(winner_ds, STROKE_WIDTH, "mark film-mark upset-winner")
        lose_svg = _svg(loser_ds, STROKE_WIDTH, "mark film-mark")
        items.append(
            f'<div class="upset-item"><div class="upset-pair">{win_svg}{lose_svg}</div>'
            f'<div class="upset-cap">vote #{pr["idx"]} ({html.escape(pr["mode"])}) &mdash; '
            f'model gave the winner P={p_winner * 100:.0f}%</div></div>')
    return f'<div class="upset-row">{"".join(items)}</div>'


def calibration_section_html(preds: list[dict], basis: PCABasis) -> str:
    chart_svg, brier = calibration_chart_html(preds)
    cal_caption = (f'<p class="chart-sub">Mean predicted vs. mean observed outcome, binned '
                  f'into {N_CAL_BINS} equal-width bins of predicted probability (dot size = '
                  'bin count); the dashed diagonal is perfect calibration. Early votes '
                  'predict exactly 0.5 -- the flat prior, before any evidence. Brier score '
                  f'(mean squared error of the sequential predictions): <b>{brier:.3f}</b>.</p>')
    gallery = upset_gallery_html(basis, preds)
    return (f'<h4>Reliability</h4>{chart_svg}{cal_caption}'
           f'<h4>Upset gallery</h4>{gallery}')


# ---------------------------------------------------------------------------
# Eigenspace scatter data (JS renders it; see the script template below) --
# seed points are shared across every size (same basis/session), the learned
# preference and the duel candidates are per size bucket.
# ---------------------------------------------------------------------------

def build_seed_points(basis: PCABasis, state: dict) -> list[dict]:
    """Session ``seed_zs`` (standardized, full K-length) -> full-K coefficient
    vectors for the scatter, labeled with ``basis.seed_files`` when the counts
    line up (a session's seeds should always match the pinned basis's, but an
    old/hand-edited session might not), else a generic "seed N"."""
    seed_zs = state.get("seed_zs") or []
    names = basis.seed_files if len(basis.seed_files) == len(seed_zs) else None
    out = []
    for i, z in enumerate(seed_zs):
        coeffs = [zk * s for zk, s in zip(z, basis.stds)]
        name = names[i] if names else f"seed {i + 1}"
        out.append({"name": name, "coeffs": _round(coeffs)})
    return out


def build_scatter_payload(model: PreferenceModel, votes: list[dict]) -> dict:
    """This size bucket's learned preference + every duel candidate it has
    been shown (both sides of every vote), for the eigenspace scatter's
    optional duel-candidate layer."""
    duels = []
    for i, v in enumerate(votes, start=1):
        duels.append({
            "idx": i,
            "mode": v.get("mode") or "unspecified",
            "winner": v["winner"],
            "a": _round(v["a_coeffs"]),
            "b": _round(v["b_coeffs"]),
        })
    return {"best": _round(model.best_coeffs()), "duels": duels}


def scatter_section_html(size: str) -> str:
    sid = _safe_id(size)
    return f"""
    <h3>Eigenspace</h3>
    <p class="chart-sub">The seed marks (grey), the population mean (open cross at the
    origin) and this bucket's learned preference (green star, <code>best_coeffs()</code>)
    projected onto two chosen components. Turn on duel candidates to also see every mark
    this bucket has been shown, colored by duel mode -- filled for the winner, hollow for
    the loser (both hollow on a tie). Hover or click a duel point to highlight its
    opponent -- a dashed link line joins the pair, both points enlarge, and every other
    duel point fades.</p>
    <div class="cluster-controls">
      <label>X axis <select id="axisX-{sid}"></select></label>
      <label>Y axis <select id="axisY-{sid}"></select></label>
      <label><input type="checkbox" id="showDuels-{sid}"> show duel candidates</label>
    </div>
    <div class="chart-box scatter-box">
      <div id="scatter-{sid}"></div>
      <div class="legend-row" id="legend-{sid}"></div>
    </div>
"""


# ---------------------------------------------------------------------------
# One size-bucket section
# ---------------------------------------------------------------------------

def build_size_section(basis: PCABasis, votes: list[dict], size: str, state: dict,
                       z_max: float, steps: int, n_components: int
                       ) -> tuple[str, dict, PreferenceModel]:
    """Returns (section_html, scatter_payload, model) -- the scatter payload is
    collected by the caller into one JSON blob shared by every size section's
    embedded JS, and the model is collected for the cross-size comparison
    section."""
    frames, model, preds = build_frames_and_model(basis, votes)
    mean_g = basis.decode([0.0] * basis.n_components)
    best_g = basis.decode(model.best_coeffs())
    delta_svg = _delta_view_svg(mean_g, best_g)

    section = f"""
  <section class="size-section" id="size-{html.escape(size)}">
    <h2>Size bucket: {html.escape(size)} <span class="n">({len(votes)} duels)</span></h2>

    <h3>Mean &rarr; current best guess</h3>
    {header_pair_html(basis, model, votes)}

    <h3>Small-size legibility</h3>
    {legibility_strip_html(basis, model)}

    <h3>B&eacute;zier delta</h3>
    <div class="delta-wrap">
      {delta_svg}
      <p class="chart-sub delta-caption">Best mark (ink) over the population mean (grey).</p>
    </div>

    <h3>Nearest seed</h3>
    {seed_proximity_html(basis, model, state)}

    <h3>Evolution over votes</h3>
    {filmstrip_html(basis, frames)}

    <h3>Per-axis z&#42; trajectories</h3>
    <p class="chart-sub">z&#42; (preferred standardized value) vs. vote count, shaded by
    its posterior spread (&plusmn;1 zstar_std); the zero line is the population mean. A
    gap means the axis had no interior peak at that checkpoint (an edge preference).</p>
    {trajectories_html(model, frames, basis)}

    <h3>When to stop voting</h3>
    {stopping_panel_html(frames, basis)}

    <h3>Per-axis utility curves</h3>
    {utility_curves_html(model, votes, basis, z_max, n_components)}

    <h3>Per-axis utility</h3>
    {utility_rows_html(basis, model, n_components, z_max, steps)}

    <h3>Model calibration</h3>
    {calibration_section_html(preds, basis)}

    {scatter_section_html(size)}
  </section>
"""
    return section, build_scatter_payload(model, votes), model


# ---------------------------------------------------------------------------
# Cross-size comparison: a document-level section (not per-bucket) comparing
# every size's learned preference, once >=2 buckets have votes. With fewer
# than 2, a one-line note takes its place -- rendered inline with the rest of
# ``build_html``'s footer material, since there's no section content to show.
# ---------------------------------------------------------------------------

def cross_size_overlay_svg(basis: PCABasis, sizes: list[str], models: dict) -> str:
    layers = []
    for i, size in enumerate(sizes):
        color = SIZE_PALETTE[i % len(SIZE_PALETTE)]
        ds = _paths_at_coeffs(basis, models[size].best_coeffs())
        layers.append("".join(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{STROKE_WIDTH}" '
            f'stroke-miterlimit="10"/>' for d in ds))
    body = "".join(layers)
    return (f'<svg viewBox="{VIEWBOX}" preserveAspectRatio="xMidYMid meet" '
          f'class="cross-overlay">{body}</svg>')


ZDOT_W = 460
ZDOT_ROW_H = 34
ZDOT_M = {"left": 56, "right": 20}


def cross_size_zstar_dotplot_html(sizes: list[str], models: dict, z_max: float) -> str:
    reports = {size: {r["axis"]: r for r in models[size].axis_report()} for size in sizes}
    axes_with_peaks = sorted({k for rep in reports.values() for k, r in rep.items() if r["peak"]})
    if not axes_with_peaks:
        return '<p class="chart-sub">No axis has an interior peak in any size bucket yet.</p>'

    height = ZDOT_ROW_H * len(axes_with_peaks) + 30
    sx = _lin_scale(-z_max, z_max, ZDOT_M["left"], ZDOT_W - ZDOT_M["right"])

    svg = []
    for t in _nice_ticks(-z_max, z_max, 5):
        px = sx(t)
        svg.append(f'<line class="grid" x1="{px:.1f}" x2="{px:.1f}" y1="10" y2="{height - 20}"/>')
        svg.append(f'<text x="{px:.1f}" y="{height - 6}" text-anchor="middle">{t:g}</text>')
    zx = sx(0.0)
    svg.append(f'<line class="zero-line" x1="{zx:.1f}" x2="{zx:.1f}" y1="10" y2="{height - 20}"/>')

    for row_i, k in enumerate(axes_with_peaks):
        y = 24 + row_i * ZDOT_ROW_H
        svg.append(f'<text x="4" y="{y + 4}" text-anchor="start" class="zdot-rowlabel">PC{k + 1}</text>')
        for i, size in enumerate(sizes):
            r = reports[size].get(k)
            if r is None or not r["peak"]:
                continue
            color = SIZE_PALETTE[i % len(SIZE_PALETTE)]
            z = max(-z_max, min(z_max, r["z_star"]))
            std = r["zstar_std"]
            px = sx(z)
            lo, hi = sx(max(-z_max, z - std)), sx(min(z_max, z + std))
            svg.append(f'<line class="zdot-err" x1="{lo:.1f}" x2="{hi:.1f}" '
                      f'y1="{y:.1f}" y2="{y:.1f}" stroke="{color}"/>')
            title = f'{html.escape(size)}: z&#42;={r["z_star"]:+.2f} &plusmn; {std:.2f}'
            svg.append(f'<circle class="zdot-pt" cx="{px:.1f}" cy="{y:.1f}" r="4" '
                      f'fill="{color}"><title>{title}</title></circle>')

    return (f'<svg width="{ZDOT_W}" height="{height}" viewBox="0 0 {ZDOT_W} {height}" '
          f'class="axis zdot-svg">{"".join(svg)}</svg>')


def cross_size_section_html(basis: PCABasis, sizes: list[str], models: dict,
                            z_max: float) -> str:
    if len(sizes) < 2:
        known = html.escape(sizes[0]) if sizes else "none"
        return (f'<p class="sub cross-note">Cross-size comparison needs votes logged in '
               f'&ge;2 size buckets (currently voted: {known}) -- once a second bucket has '
               'votes, this section will overlay every bucket\'s best guess and compare '
               'z&#42; per axis across sizes.</p>')

    legend = "".join(
        f'<span><span class="swatch" style="background:{SIZE_PALETTE[i % len(SIZE_PALETTE)]}">'
        f'</span>{html.escape(size)}</span>'
        for i, size in enumerate(sizes))
    overlay = cross_size_overlay_svg(basis, sizes, models)
    dotplot = cross_size_zstar_dotplot_html(sizes, models, z_max)
    return f"""
  <section class="size-section cross-section">
    <h2>Cross-size comparison</h2>
    <h3>Best guess per size, overlaid</h3>
    <div class="delta-wrap">
      {overlay}
      <div class="legend-row cross-legend">{legend}</div>
    </div>
    <h3>z&#42; per axis, across sizes</h3>
    <p class="chart-sub">One dot per size per axis with an interior peak in that bucket
    (error bars &plusmn;1 zstar_std). Agreement across sizes means the preference doesn't
    depend on display size; divergence means it does.</p>
    {dotplot}
  </section>
"""


# ---------------------------------------------------------------------------
# Whole document
# ---------------------------------------------------------------------------

def build_html(state: dict, basis: PCABasis, sections: list[str],
              sizes: list[str], by_size: dict, z_max: float,
              scatter_payloads: dict, cross_html: str) -> str:
    seeds = html.escape(", ".join(basis.seed_files))
    total_votes = sum(len(by_size[s]) for s in sizes)
    nav = ""
    if len(sizes) > 1:
        links = " &middot; ".join(
            f'<a href="#size-{html.escape(s)}">{html.escape(s)} ({len(by_size[s])})</a>'
            for s in sizes)
        nav = f'<p class="sizenav">Jump to: {links}</p>'

    total_var = sum(basis.eigenvalues) or 1.0
    global_data = {
        "nComp": basis.n_components,
        "varPct": _round([basis.eigenvalues[k] / total_var * 100.0
                          for k in range(basis.n_components)]),
        "seeds": build_seed_points(basis, state),
    }
    script = _SCRIPT_TEMPLATE
    script = script.replace("__GLOBAL__", json.dumps(global_data, separators=(",", ":")).replace("</", "<\\/"))
    script = script.replace("__SIZE_DATA__", json.dumps(scatter_payloads, separators=(",", ":")).replace("</", "<\\/"))
    script = script.replace("__MODE_COLORS__", json.dumps(MODE_COLORS))
    script = script.replace("__SIZES__", json.dumps(sizes))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Maker's Mark preferences</title>
<style>
  :root {{ --ink:#231f20; }}
  body {{ font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif; color:var(--ink);
         margin:0; padding:24px 28px; background:#fafafa; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  h2 {{ font-size:18px; margin:34px 0 2px; padding-top:14px; border-top:1px solid #ddd; }}
  h2 .n {{ font-weight:400; color:#888; font-size:14px; }}
  h3 {{ font-size:14px; margin:16px 0 8px; color:#444; }}
  .sub {{ color:#666; margin:0 0 16px; max-width:76ch; }}
  .warn {{ color:#b8860b; margin:6px 0; }}
  .sizenav {{ color:#555; font-size:13px; margin:0 0 18px; }}
  .sizenav a {{ color:#1a5fb4; text-decoration:none; margin-right:4px; }}
  .sizenav a:hover {{ text-decoration:underline; }}

  .summary {{ display:flex; gap:24px; align-items:center; margin:0 0 6px; flex-wrap:wrap; }}
  .summary figure {{ margin:0; text-align:center; }}
  .summary figcaption {{ color:#666; font-size:12px; margin-top:4px; }}
  .big {{ width:130px; height:148px; background:#fff; border:1px solid #ddd;
          border-radius:6px; display:block; }}
  .arrow {{ font-size:24px; color:#1a7f37; }}
  .summary-stats {{ font-size:13px; color:#444; }}
  .stat-n {{ margin-bottom:2px; }}
  .stat-modes {{ color:#777; font-size:12px; }}

  .delta-wrap {{ display:flex; align-items:flex-start; gap:16px; flex-wrap:wrap; }}
  .delta-wrap .mark.big {{ width:220px; height:250px; }}
  .chart-sub {{ font-size:12px; color:#777; max-width:52ch; margin:4px 0; }}

  .film-row {{ display:flex; flex-wrap:wrap; gap:10px; }}
  .film-cell {{ text-align:center; }}
  .film-mark {{ width:64px; height:72px; background:#fff; border:1px solid #eee;
               border-radius:4px; }}
  .film-label {{ font-size:11px; color:#777; margin-top:2px; }}

  .traj-grid {{ display:flex; flex-wrap:wrap; gap:14px; }}
  .traj-item {{ background:#fff; border:1px solid #eee; border-radius:6px; padding:8px 10px; }}
  .traj-title {{ font-weight:700; font-size:12px; }}
  .traj-sub {{ color:#c0392b; font-size:10px; margin-bottom:2px; }}
  .traj-empty {{ width:{TRAJ_W}px; height:{TRAJ_H}px; display:flex; align-items:center;
                justify-content:center; text-align:center; color:#999; font-size:11px; }}
  .axis text {{ font:10px -apple-system,Segoe UI,Roboto,sans-serif; fill:#898781;
               font-variant-numeric:tabular-nums; }}
  .axis .grid {{ stroke:#e1e0d9; stroke-width:1; }}
  .axis .baseline {{ stroke:#c3c2b7; stroke-width:1; }}
  .axis .zero-line {{ stroke:#b6b4a8; stroke-width:1; stroke-dasharray:3,2; }}
  .traj-band {{ fill:{TRAJ_COLOR}; opacity:0.15; stroke:none; }}
  .traj-line {{ fill:none; stroke:{TRAJ_COLOR}; stroke-width:2; }}
  .traj-dot {{ fill:{TRAJ_COLOR}; stroke:#fff; stroke-width:1.5; }}
  .traj-err {{ stroke:{TRAJ_COLOR}; stroke-width:2; }}
  .traj-final {{ font:700 11px -apple-system,Segoe UI,Roboto,sans-serif; fill:{TRAJ_COLOR};
                font-variant-numeric:tabular-nums; }}

  .rowlabel {{ text-align:left; padding:0 10px 0 2px; vertical-align:middle; white-space:nowrap; }}
  .settle-tag {{ display:inline-block; font-size:10px; padding:1px 6px; border-radius:9px;
                margin-top:2px; }}
  .settle-tag.settled {{ background:#e6f4ea; color:#1a7f37; }}
  .settle-tag.unsettled {{ background:#f4ede3; color:#a06a1f; }}
  .mark {{ display:block; width:82px; height:92px; margin:2px auto 0; }}

  table {{ border-collapse:collapse; }}
  thead th {{ font-weight:600; color:#555; font-size:12px; padding:4px 6px; text-align:center; }}
  .util-label {{ min-width:150px; }}
  .util-label .pc {{ font-weight:700; font-size:15px; }}
  .util-label .star {{ color:#1a7f37; font-size:12px; font-weight:600; }}
  .util-label .kind {{ font-size:11px; color:#888; }}
  .util-label .kind.edge {{ color:#c0392b; }}
  .cell {{ vertical-align:middle; text-align:center; border:1px solid #eee; padding:4px; }}
  .cell.star {{ outline:2px solid #1a7f37; outline-offset:-2px; }}
  .cell.overlay {{ padding:6px; }}
  .util-overlay-svg {{ width:150px; height:170px; background:#fff; }}
  .bar {{ height:26px; display:flex; align-items:flex-end; justify-content:center; }}
  .bar i {{ display:block; width:60%; background:#1a7f37; opacity:.55; }}
  .cell.profile {{ background:#f4fbf6; border-left:2px solid #cfe9d8; }}
  .cell.profile .ptag {{ font-size:11px; color:#1a7f37; padding:2px 0 4px; }}
  .ramp-legend {{ display:flex; align-items:center; gap:8px; margin-top:10px;
                 font-size:11px; color:#666; max-width:60ch; }}
  .ramp-swatch {{ display:inline-block; width:80px; height:10px; border-radius:5px;
                 flex:none; }}

  .cluster-controls {{ display:flex; flex-wrap:wrap; gap:20px; align-items:center;
                       margin:10px 0 16px; font-size:13px; color:#444; }}
  .cluster-controls label {{ display:flex; align-items:center; gap:6px; }}
  .cluster-controls select {{ font:inherit; }}
  .chart-box {{ background:#fff; border:1px solid #eee; border-radius:6px; padding:14px 16px;
               display:inline-block; }}
  .legend-row {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:10px; font-size:11px; color:#555; }}
  .legend-row .swatch {{ display:inline-block; width:10px; height:10px; border-radius:2px;
                         vertical-align:-1px; margin-right:4px; }}
  .seed-swatch {{ background:{SEED_COLOR}; }}
  .mean-swatch {{ background:none; border:1.5px solid {MEAN_MARKER_COLOR}; }}
  .best-swatch {{ background:{BEST_COLOR}; clip-path:polygon(50% 0%,61% 35%,98% 35%,68% 57%,
                 79% 91%,50% 70%,21% 91%,32% 57%,2% 35%,39% 35%); }}
  .axis .seed-point {{ fill:{SEED_COLOR}; stroke:#fff; stroke-width:1.5; cursor:default; }}
  .axis .mean-point line {{ stroke:{MEAN_MARKER_COLOR}; stroke-width:2; }}
  .axis .best-point {{ fill:{BEST_COLOR}; stroke:#fff; stroke-width:1.5; }}
  .axis .duel-point {{ cursor:pointer; pointer-events:all; transition:opacity .12s; }}
  .axis .duel-link {{ opacity:0; stroke-width:1.5; stroke-dasharray:4,3; pointer-events:none;
                      transition:opacity .12s; }}
  .axis.highlighting .duel-point {{ opacity:.15; }}
  .axis.highlighting .duel-point.active {{ opacity:1; r:6; }}
  .axis.highlighting .duel-link.active {{ opacity:.85; }}

  h4 {{ font-size:12.5px; margin:12px 0 6px; color:#555; }}

  .legib-row {{ display:flex; flex-wrap:wrap; gap:14px; align-items:flex-end; margin:6px 0; }}
  .legib-cell {{ text-align:center; }}
  .legib-mark {{ background:#fff; border:1px solid #eee; border-radius:3px; overflow:hidden; }}
  .legib-svg {{ width:100%; height:100%; display:block; }}
  .legib-cap {{ font-size:11px; color:#777; margin-top:3px; }}

  .mark.mid {{ width:140px; height:158px; }}
  .seedprox-row {{ display:flex; flex-wrap:wrap; gap:16px; margin:6px 0; }}
  .seedprox-item {{ text-align:center; }}
  .seedprox-label {{ font-size:12px; color:#555; margin-top:4px; }}

  .ucurve-map {{ fill:none; stroke:{BEST_COLOR}; stroke-width:2.5; }}
  .ucurve-draw {{ fill:none; stroke:{BEST_COLOR}; stroke-width:1; opacity:0.14; }}
  .ucurve-zstar {{ stroke:#888; stroke-width:1.3; stroke-dasharray:3,2; }}
  .tick-win {{ stroke:{INK_STROKE}; stroke-width:2.2; }}
  .tick-lose {{ stroke:#b0b0b0; stroke-width:1.6; }}
  .tick-tie {{ stroke:#b0b0b0; stroke-width:1.6; }}

  .stop-line {{ fill:none; stroke:{TRAJ_COLOR}; stroke-width:2; }}
  .stop-dot {{ fill:{TRAJ_COLOR}; stroke:#fff; stroke-width:1.3; }}

  .cal-diag {{ stroke:#bbb; stroke-width:1.5; stroke-dasharray:4,3; }}
  .cal-dot {{ fill:{TRAJ_COLOR}; opacity:.75; stroke:#fff; stroke-width:1; }}

  .upset-row {{ display:flex; flex-wrap:wrap; gap:14px; margin:6px 0; }}
  .upset-item {{ text-align:center; background:#fff; border:1px solid #eee; border-radius:6px;
                padding:8px; }}
  .upset-pair {{ display:flex; gap:6px; justify-content:center; }}
  .upset-cap {{ font-size:11px; color:#666; margin-top:4px; max-width:220px; }}
  .upset-winner {{ outline:2px solid {BEST_COLOR}; outline-offset:-2px; border-radius:3px; }}

  .cross-overlay {{ display:block; width:220px; height:250px; background:#fff;
                    border:1px solid #ddd; border-radius:6px; }}
  .cross-legend {{ align-self:center; }}
  .zdot-svg .zdot-rowlabel {{ fill:#444; font-weight:600; }}
  .zdot-err {{ stroke-width:2.5; }}
  .zdot-pt {{ stroke:#fff; stroke-width:1.3; }}
  .cross-note {{ color:#666; }}
</style></head>
<body>
  <h1>Maker&rsquo;s Mark preferences</h1>
  <p class="sub">One Bayesian peaked model per size bucket, fitted to <b>{total_votes}</b>
  logged duels total across {basis.n_seeds} seeds ({basis.n_components} components, every
  axis learned and varied -- the hybrid scheduler just weights duels toward the
  highest-variance ones). Each section below is one size bucket.</p>
  {nav}
  {"".join(sections)}
  {cross_html}
  <p class="sub" style="margin-top:20px">
    <b>peak</b> = an interior sweet spot (utility falls off on both sides);
    <b class="warn" style="color:#c0392b">edge</b> = preference keeps rising to the
    &plusmn;{z_max:g}&sigma; bound, no interior optimum yet.
    &nbsp;&middot;&nbsp; seeds: {seeds}
  </p>
  <script>
{script}
  </script>
</body></html>
"""


# ---------------------------------------------------------------------------
# Eigenspace scatter script (embedded, no server -- reuses sample_display.py's
# JS helper patterns: escapeHtml, niceTicks, scaleLinear, .axis/.grid/.baseline
# CSS classes).
# ---------------------------------------------------------------------------

_SCRIPT_TEMPLATE = r"""
const GLOBAL = __GLOBAL__;
const SIZE_DATA = __SIZE_DATA__;
const MODE_COLORS = __MODE_COLORS__;
const MODE_ORDER = Object.keys(MODE_COLORS);
const SIZES = __SIZES__;

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

function niceTicks(lo, hi, count) {
  if (lo === hi) { lo -= 1; hi += 1; }
  const span = hi - lo;
  const step0 = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  let step;
  if (norm < 1.5) step = mag;
  else if (norm < 3) step = 2 * mag;
  else if (norm < 7) step = 5 * mag;
  else step = 10 * mag;
  const start = Math.ceil(lo / step) * step;
  const ticks = [];
  for (let v = start; v <= hi + 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

function scaleLinear(d0, d1, r0, r1) {
  const m = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  return v => r0 + (v - d0) * m;
}

function starPath(cx, cy, r) {
  const pts = [];
  for (let i = 0; i < 10; i++) {
    const rad = i % 2 === 0 ? r : r * 0.45;
    const ang = -Math.PI / 2 + i * Math.PI / 5;
    pts.push([cx + rad * Math.cos(ang), cy + rad * Math.sin(ang)]);
  }
  return "M" + pts.map(p => p[0].toFixed(1) + "," + p[1].toFixed(1)).join("L") + "Z";
}

const SCATTER_W = 460, SCATTER_H = 380, SCATTER_M = {left: 46, right: 16, top: 16, bottom: 40};

// Which duel (by its "idx") is pinned, per size section -- reset on every
// render (axis change / show-duels toggle), so a stale pin never survives
// a redraw of a different projection.
const pinnedDuel = {};

function renderScatterFor(size) {
  const sid = size.replace(/[^A-Za-z0-9_-]/g, "_");
  const xk = +document.getElementById("axisX-" + sid).value;
  const yk = +document.getElementById("axisY-" + sid).value;
  const showDuels = document.getElementById("showDuels-" + sid).checked;
  const data = SIZE_DATA[size];
  pinnedDuel[sid] = null;

  const seedPts = GLOBAL.seeds.map(s => [s.coeffs[xk], s.coeffs[yk]]);
  const bestPt = [data.best[xk], data.best[yk]];
  const meanPt = [0, 0];
  const duelPts = [];
  if (showDuels) {
    data.duels.forEach(d => { duelPts.push([d.a[xk], d.a[yk]]); duelPts.push([d.b[xk], d.b[yk]]); });
  }

  const allX = seedPts.map(p => p[0]).concat([bestPt[0], meanPt[0]], duelPts.map(p => p[0]));
  const allY = seedPts.map(p => p[1]).concat([bestPt[1], meanPt[1]], duelPts.map(p => p[1]));
  const xPad = (Math.max(...allX) - Math.min(...allX)) * 0.15 || 1;
  const yPad = (Math.max(...allY) - Math.min(...allY)) * 0.15 || 1;
  const x0 = Math.min(...allX) - xPad, x1 = Math.max(...allX) + xPad;
  const y0 = Math.min(...allY) - yPad, y1 = Math.max(...allY) + yPad;
  const sx = scaleLinear(x0, x1, SCATTER_M.left, SCATTER_W - SCATTER_M.right);
  const sy = scaleLinear(y0, y1, SCATTER_H - SCATTER_M.bottom, SCATTER_M.top);

  let svg = "";
  niceTicks(x0, x1, 5).forEach(t => {
    const px = sx(t);
    svg += `<line class="grid" x1="${px}" x2="${px}" y1="${SCATTER_M.top}" y2="${SCATTER_H - SCATTER_M.bottom}"/>`;
    svg += `<text x="${px}" y="${SCATTER_H - SCATTER_M.bottom + 14}" text-anchor="middle">${t}</text>`;
  });
  niceTicks(y0, y1, 5).forEach(t => {
    const py = sy(t);
    svg += `<line class="grid" x1="${SCATTER_M.left}" x2="${SCATTER_W - SCATTER_M.right}" y1="${py}" y2="${py}"/>`;
    svg += `<text x="${SCATTER_M.left - 8}" y="${py + 3}" text-anchor="end">${t}</text>`;
  });
  svg += `<line class="baseline" x1="${SCATTER_M.left}" x2="${SCATTER_W - SCATTER_M.right}" y1="${SCATTER_H - SCATTER_M.bottom}" y2="${SCATTER_H - SCATTER_M.bottom}"/>`;
  svg += `<line class="baseline" x1="${SCATTER_M.left}" x2="${SCATTER_M.left}" y1="${SCATTER_M.top}" y2="${SCATTER_H - SCATTER_M.bottom}"/>`;
  svg += `<text class="label" x="${(SCATTER_M.left + SCATTER_W - SCATTER_M.right) / 2}" y="${SCATTER_H - 6}" text-anchor="middle">PC${xk + 1} coefficient</text>`;
  svg += `<text class="label" x="${-(SCATTER_M.top + SCATTER_H - SCATTER_M.bottom) / 2}" y="12" text-anchor="middle" transform="rotate(-90)">PC${yk + 1} coefficient</text>`;

  if (showDuels) {
    // Link lines first, so they paint *under* the point circles drawn next.
    data.duels.forEach(d => {
      const color = MODE_COLORS[d.mode] || MODE_COLORS.unspecified;
      const ax = sx(d.a[xk]), ay = sy(d.a[yk]);
      const bx = sx(d.b[xk]), by = sy(d.b[yk]);
      svg += `<line class="duel-link" data-di="${d.idx}" x1="${ax.toFixed(1)}" y1="${ay.toFixed(1)}" ` +
             `x2="${bx.toFixed(1)}" y2="${by.toFixed(1)}" stroke="${color}"/>`;
    });
    data.duels.forEach(d => {
      [["a", d.a], ["b", d.b]].forEach(([side, c]) => {
        const px = sx(c[xk]), py = sy(c[yk]);
        const color = MODE_COLORS[d.mode] || MODE_COLORS.unspecified;
        const isWinner = d.winner === side;
        const filled = d.winner !== "tie" && isWinner;
        const outcome = d.winner === "tie" ? "tie" : (isWinner ? "winner" : "loser");
        const title = `vote #${d.idx}, mode ${d.mode}, ${outcome}\nPC${xk + 1}=${c[xk].toFixed(3)}, PC${yk + 1}=${c[yk].toFixed(3)}`;
        svg += `<circle class="duel-point" data-di="${d.idx}" cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="4" ` +
               `fill="${filled ? color : "none"}" stroke="${color}" stroke-width="1.5">` +
               `<title>${escapeHtml(title)}</title></circle>`;
      });
    });
  }

  seedPts.forEach((p, i) => {
    const px = sx(p[0]), py = sy(p[1]);
    const title = `${GLOBAL.seeds[i].name}\nPC${xk + 1}=${p[0].toFixed(3)}, PC${yk + 1}=${p[1].toFixed(3)}`;
    svg += `<circle class="seed-point" cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="4">` +
           `<title>${escapeHtml(title)}</title></circle>`;
  });

  {
    const px = sx(meanPt[0]), py = sy(meanPt[1]);
    svg += `<g class="mean-point"><line x1="${px - 6}" y1="${py}" x2="${px + 6}" y2="${py}"/>` +
           `<line x1="${px}" y1="${py - 6}" x2="${px}" y2="${py + 6}"/>` +
           `<title>population mean</title></g>`;
  }

  {
    const px = sx(bestPt[0]), py = sy(bestPt[1]);
    const title = `learned preference (best_coeffs)\nPC${xk + 1}=${bestPt[0].toFixed(3)}, PC${yk + 1}=${bestPt[1].toFixed(3)}`;
    svg += `<path class="best-point" d="${starPath(px, py, 10)}"><title>${escapeHtml(title)}</title></path>`;
  }

  const scatterEl = document.getElementById("scatter-" + sid);
  scatterEl.innerHTML =
    `<svg width="${SCATTER_W}" height="${SCATTER_H}" viewBox="0 0 ${SCATTER_W} ${SCATTER_H}" class="axis">${svg}</svg>`;

  if (showDuels) {
    const svgEl = scatterEl.querySelector("svg");
    const points = svgEl.querySelectorAll(".duel-point");
    const links = svgEl.querySelectorAll(".duel-link");

    const applyHighlight = di => {
      if (di === null) {
        svgEl.classList.remove("highlighting");
        points.forEach(p => p.classList.remove("active"));
        links.forEach(l => l.classList.remove("active"));
        return;
      }
      svgEl.classList.add("highlighting");
      points.forEach(p => p.classList.toggle("active", +p.dataset.di === di));
      links.forEach(l => l.classList.toggle("active", +l.dataset.di === di));
    };

    points.forEach(p => {
      p.addEventListener("mouseenter", () => {
        if (pinnedDuel[sid] === null) applyHighlight(+p.dataset.di);
      });
      p.addEventListener("mouseleave", () => {
        if (pinnedDuel[sid] === null) applyHighlight(null);
      });
      p.addEventListener("click", () => {
        const di = +p.dataset.di;
        pinnedDuel[sid] = pinnedDuel[sid] === di ? null : di;
        applyHighlight(pinnedDuel[sid]);
      });
    });
  }

  const legendBits = [
    '<span><span class="swatch seed-swatch"></span>seed</span>',
    '<span><span class="swatch mean-swatch"></span>population mean</span>',
    '<span><span class="swatch best-swatch"></span>learned preference</span>',
  ];
  if (showDuels) {
    MODE_ORDER.forEach(m => {
      legendBits.push(`<span><span class="swatch" style="background:${MODE_COLORS[m]}"></span>${m}</span>`);
    });
    legendBits.push("<span>filled = winner &middot; hollow = loser/tie</span>");
    legendBits.push("<span>hover or click a duel point to highlight its opponent</span>");
  }
  document.getElementById("legend-" + sid).innerHTML = legendBits.join("");
}

function initScatter(size) {
  const sid = size.replace(/[^A-Za-z0-9_-]/g, "_");
  const axisX = document.getElementById("axisX-" + sid);
  const axisY = document.getElementById("axisY-" + sid);
  for (let k = 0; k < GLOBAL.nComp; k++) {
    const pct = GLOBAL.varPct[k].toFixed(1);
    axisX.add(new Option(`PC${k + 1} (${pct}% var)`, k, k === 0, k === 0));
    axisY.add(new Option(`PC${k + 1} (${pct}% var)`, k, k === 1, k === 1));
  }
  axisX.addEventListener("change", () => renderScatterFor(size));
  axisY.addEventListener("change", () => renderScatterFor(size));
  document.getElementById("showDuels-" + sid).addEventListener("change", () => renderScatterFor(size));
  renderScatterFor(size);
}

SIZES.forEach(initScatter);
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="preference_results.html",
                    help="output HTML file (default preference_results.html)")
    ap.add_argument("-n", "--components", type=int, default=999,
                    help="max eigen-axes to show in the utility-rows view (default: all)")
    ap.add_argument("--data-dir", default=str(DATA_DIR),
                    help="session directory (default pref_data/)")
    ap.add_argument("--z-max", type=float, default=2.5,
                    help="std-devs to walk each side of the mean in the utility-rows view")
    ap.add_argument("--steps", type=int, default=9,
                    help="cells per row across the utility-rows walk (odd; mean centered)")
    ap.add_argument("--size", action="append", default=None,
                    help="only render these size buckets (repeatable or comma-separated); "
                         "default: every bucket with >=1 vote")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    state, basis, votes = load_session(data_dir)
    # A session's "n_active" (if present at all -- old sessions may predate
    # the key) is meaningless here: every model this file builds learns/varies
    # all components (see build_frames_and_model), matching
    # preference_model.py's current philosophy.
    default_size = state.get("default_size", DEFAULT_SIZE)

    by_size = group_by_size(votes, default_size)
    order = size_order(state, by_size)

    if args.size:
        requested = []
        for s in args.size:
            requested += [x.strip() for x in s.split(",") if x.strip()]
        missing = [s for s in requested if not by_size.get(s)]
        if missing:
            print(f"note: no votes logged yet for: {', '.join(missing)}")
        sizes_to_render = [s for s in order if s in requested and by_size.get(s)]
    else:
        sizes_to_render = [s for s in order if by_size.get(s)]

    if not sizes_to_render:
        raise SystemExit("no size bucket has any votes yet -- vote a bit with "
                         "preference_server.py first")

    sections = []
    scatter_payloads = {}
    models = {}
    for size in sizes_to_render:
        section, payload, model = build_size_section(basis, by_size[size], size, state,
                                                      args.z_max, args.steps, args.components)
        sections.append(section)
        scatter_payloads[size] = payload
        models[size] = model

    cross_html = cross_size_section_html(basis, sizes_to_render, models, args.z_max)

    doc = build_html(state, basis, sections, sizes_to_render, by_size, args.z_max,
                     scatter_payloads, cross_html)
    Path(args.out).write_text(doc, encoding="utf-8")

    counts = ", ".join(f"{s}: {len(by_size[s])}" for s in sizes_to_render)
    print(f"rendered {len(sizes_to_render)} size bucket(s) ({counts}); wrote {args.out}")


if __name__ == "__main__":
    main()
