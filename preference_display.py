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
from preference_model import WINNER_Y, PreferenceModel

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
    session is ignored here."""
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
    ci = 0
    for i, v in enumerate(votes, start=1):
        model.observe(v["a_coeffs"], v["b_coeffs"], v["winner"])
        if ci < len(checkpoints) and i == checkpoints[ci]:
            snapshot(i)
            ci += 1
    return frames, model


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

def build_size_section(basis: PCABasis, votes: list[dict], size: str,
                       z_max: float, steps: int, n_components: int) -> tuple[str, dict]:
    """Returns (section_html, scatter_payload) -- the scatter payload is
    collected by the caller into one JSON blob shared by every size section's
    embedded JS."""
    frames, model = build_frames_and_model(basis, votes)
    mean_g = basis.decode([0.0] * basis.n_components)
    best_g = basis.decode(model.best_coeffs())
    delta_svg = _delta_view_svg(mean_g, best_g)

    section = f"""
  <section class="size-section" id="size-{html.escape(size)}">
    <h2>Size bucket: {html.escape(size)} <span class="n">({len(votes)} duels)</span></h2>

    <h3>Mean &rarr; current best guess</h3>
    {header_pair_html(basis, model, votes)}

    <h3>B&eacute;zier delta</h3>
    <div class="delta-wrap">
      {delta_svg}
      <p class="chart-sub delta-caption">Best mark (ink) over the population mean (grey).</p>
    </div>

    <h3>Evolution over votes</h3>
    {filmstrip_html(basis, frames)}

    <h3>Per-axis z&#42; trajectories</h3>
    <p class="chart-sub">z&#42; (preferred standardized value) vs. vote count, shaded by
    its posterior spread (&plusmn;1 zstar_std); the zero line is the population mean. A
    gap means the axis had no interior peak at that checkpoint (an edge preference).</p>
    {trajectories_html(model, frames, basis)}

    <h3>Per-axis utility</h3>
    {utility_rows_html(basis, model, n_components, z_max, steps)}

    {scatter_section_html(size)}
  </section>
"""
    return section, build_scatter_payload(model, votes)


# ---------------------------------------------------------------------------
# Whole document
# ---------------------------------------------------------------------------

def build_html(state: dict, basis: PCABasis, sections: list[str],
              sizes: list[str], by_size: dict, z_max: float,
              scatter_payloads: dict) -> str:
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
</style></head>
<body>
  <h1>Maker&rsquo;s Mark preferences</h1>
  <p class="sub">One Bayesian peaked model per size bucket, fitted to <b>{total_votes}</b>
  logged duels total across {basis.n_seeds} seeds ({basis.n_components} components, every
  axis learned and varied -- the hybrid scheduler just weights duels toward the
  highest-variance ones). Each section below is one size bucket.</p>
  {nav}
  {"".join(sections)}
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
    for size in sizes_to_render:
        section, payload = build_size_section(basis, by_size[size], size, args.z_max,
                                              args.steps, args.components)
        sections.append(section)
        scatter_payloads[size] = payload

    doc = build_html(state, basis, sections, sizes_to_render, by_size, args.z_max,
                     scatter_payloads)
    Path(args.out).write_text(doc, encoding="utf-8")

    counts = ", ".join(f"{s}: {len(by_size[s])}" for s in sizes_to_render)
    print(f"rendered {len(sizes_to_render)} size bucket(s) ({counts}); wrote {args.out}")


if __name__ == "__main__":
    main()
