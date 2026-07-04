"""Visualize what ``preference_server.py`` learned about the eigenshape space.

``eigen_display.py`` shows what each PCA axis *does*; this shows which values
along those axes you *prefer* -- and, since a model is fitted **per size
bucket**, how that preference looks at each size. It reads a session
(``pref_data/session.json`` for the pinned basis and ``n_active``) and its
logged duels (``pref_data/votes.jsonl``), refits one ``PreferenceModel`` per
size bucket, and writes a self-contained ``preference_results.html`` with one
section per size (default: every bucket with at least one vote, in session
order; narrow it with ``--size``). Each section has, top to bottom:

* a **header pair** -- the population mean next to the model's current best
  guess (``PreferenceModel.best_coeffs()``), with that bucket's vote/tie count
  and a breakdown by duel mode (axis/blend/confirm);
* a **Bezier delta view** -- the best mark in ink over the mean in grey, with
  an arrow per skeleton node from its mean position to its best-mark position
  (small displacements are scaled up for visibility, noted in the caption);
* an **evolution filmstrip** -- the model refit on prefixes of that bucket's
  vote log (checkpoints every ``max(5, n//12)`` votes, incrementally --
  ``observe()`` warm-starts IRLS from the last fit, so this doesn't refit from
  scratch), ``best_coeffs()`` decoded at each checkpoint;
* **per-axis z\\* trajectories** -- small-multiple line charts of each active
  axis's preferred z\\* against vote count (shaded by its posterior
  ``zstar_std``, a zero line marking the mean, gaps where the axis had no
  interior peak yet), for the active axes (or the top ~8 by final utility
  span if there are many); and
* **per-axis utility rows** -- the mark stepped across each axis, cells tinted
  by utility, z\\* outlined, now labeled with its posterior spread
  (``z* = 0.8 +/- 0.3``) and a settled/unsettled tag.

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
import statistics
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
# Reuses sample_display.py's validated categorical blue (dataviz skill palette)
# for "this is a delta/trend", distinct from this file's existing green=peak /
# red=edge semantics.
ARROW_COLOR = "#2a78d6"
TRAJ_COLOR = "#2a78d6"
ARROW_SCALE = 3.0            # amplify node-displacement arrows for visibility
ARROW_SCALE_THRESHOLD = 3.0  # px; below this typical displacement, scale up
SETTLED_ZSTD_STD = 0.35      # zstar_std under this (+ an interior peak) => "settled"


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


def _util_bg(norm: float) -> str:
    """norm in [0,1] -> white (low utility) to green (high utility)."""
    r = round((1 - norm) * 255 + norm * 0x1a)
    g = round((1 - norm) * 255 + norm * 0x7f)
    b = round((1 - norm) * 255 + norm * 0x37)
    return "#%02x%02x%02x" % (r, g, b)


def _all_nodes(g) -> list[tuple[float, float]]:
    """Every skeleton ("node" gene) point across all three paths, in a fixed
    order -- ``PathGene.nodes`` *is* the absolute-coordinate node-gene values
    (see genome.py's Gene/PathGene), so no extra decoding is needed."""
    pts = []
    for pid in PATH_ORDER:
        pts.extend(g.paths[pid].nodes)
    return pts


def _delta_view_svg(mean_g, best_g) -> tuple[str, float, float]:
    """Mean (grey) under best (ink), with an arrow per node from its mean
    position to its best-mark position. Returns (svg, arrow_scale, median_px)."""
    mean_pts = _all_nodes(mean_g)
    best_pts = _all_nodes(best_g)
    disps = [math.hypot(bx - mx, by - my) for (mx, my), (bx, by) in zip(mean_pts, best_pts)]
    median = statistics.median(disps) if disps else 0.0
    scale = ARROW_SCALE if median < ARROW_SCALE_THRESHOLD else 1.0

    arrows = []
    for (mx, my), (bx, by) in zip(mean_pts, best_pts):
        ex, ey = mx + (bx - mx) * scale, my + (by - my) * scale
        if math.hypot(ex - mx, ey - my) < 0.05:
            continue
        arrows.append(f'<circle cx="{mx:.2f}" cy="{my:.2f}" r="0.8" fill="{ARROW_COLOR}"/>')
        arrows.append(f'<line x1="{mx:.2f}" y1="{my:.2f}" x2="{ex:.2f}" y2="{ey:.2f}" '
                      f'stroke="{ARROW_COLOR}" stroke-width="0.9" '
                      f'marker-end="url(#deltaArrow)"/>')

    mean_ds = [mean_g.paths[pid].to_d() for pid in PATH_ORDER]
    best_ds = [best_g.paths[pid].to_d() for pid in PATH_ORDER]
    body = (
        "".join(f'<path d="{d}" fill="none" stroke="{MEAN_STROKE}" '
               f'stroke-width="{STROKE_WIDTH}" stroke-miterlimit="10"/>' for d in mean_ds)
        + "".join(f'<path d="{d}" fill="none" stroke="{INK_STROKE}" '
                 f'stroke-width="{STROKE_WIDTH}" stroke-miterlimit="10"/>' for d in best_ds)
        + "".join(arrows)
    )
    svg = (f'<svg viewBox="{VIEWBOX}" preserveAspectRatio="xMidYMid meet" '
          f'class="mark big">{body}</svg>')
    return svg, scale, median


# ---------------------------------------------------------------------------
# Evolution filmstrip: one model, observed incrementally, snapshotted
# ---------------------------------------------------------------------------

def build_frames_and_model(basis: PCABasis, votes: list[dict], n_active: int,
                           seed: int = 0):
    """Refit on checkpoints of ``votes`` without ever refitting from scratch:
    one ``PreferenceModel``, ``observe()``-d one vote at a time (IRLS warm-
    starts from the previous fit), snapshotting at each checkpoint. The model
    returned is therefore also the final, fully-observed model for this size
    bucket -- callers reuse it for the header/delta/utility-rows views instead
    of building a second one."""
    model = PreferenceModel(basis.stds, n_active=n_active, rng=random.Random(seed))
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
# Per-axis utility rows (same walk-and-tint view as before, now per size and
# labeled with the axis's posterior spread + a settled/unsettled tag)
# ---------------------------------------------------------------------------

def utility_rows_html(basis: PCABasis, model: PreferenceModel, n_components: int,
                      z_max: float, steps: int) -> str:
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
    ) + "<th>profile</th>"

    rows = []
    for r in report[:n_show]:
        k = r["axis"]
        std = basis.stds[k]
        lin, quad = r["lin"], r["quad"]
        z_star = r["z_star"]
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
        cells.append(
            f'<td class="cell profile">{_svg(star_ds, STROKE_WIDTH)}'
            f'<div class="ptag">{kind} @ {z_star:+.2f}σ</div></td>')

        settled = r["peak"] and r["zstar_std"] < SETTLED_ZSTD_STD
        settle_cls = "settled" if settled else "unsettled"
        settle_text = "settled" if settled else "unsettled"
        label = (f'<div class="pc">PC{k + 1}</div>'
                 f'<div class="star">z* = {z_star:+.2f} &plusmn; {r["zstar_std"]:.2f}</div>'
                 f'<div class="kind {kind}">{kind}</div>'
                 f'<div class="settle-tag {settle_cls}">{settle_text}</div>')
        rows.append(f'<tr><th class="rowlabel">{label}</th>{"".join(cells)}</tr>')

    return f"""<table>
    <thead><tr><th></th>{col_head}</tr></thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>"""


# ---------------------------------------------------------------------------
# One size-bucket section
# ---------------------------------------------------------------------------

def build_size_section(basis: PCABasis, votes: list[dict], size: str, n_active: int,
                       z_max: float, steps: int, n_components: int) -> str:
    frames, model = build_frames_and_model(basis, votes, n_active)
    mean_g = basis.decode([0.0] * basis.n_components)
    best_g = basis.decode(model.best_coeffs())
    delta_svg, arrow_scale, median_px = _delta_view_svg(mean_g, best_g)
    scale_note = (f" Arrows are scaled &times;{arrow_scale:g} for visibility "
                 f"(typical node displacement &asymp; {median_px:.2f}px)."
                 if arrow_scale != 1.0 else "")

    return f"""
  <section class="size-section" id="size-{html.escape(size)}">
    <h2>Size bucket: {html.escape(size)} <span class="n">({len(votes)} duels)</span></h2>

    <h3>Mean &rarr; current best guess</h3>
    {header_pair_html(basis, model, votes)}

    <h3>B&eacute;zier delta</h3>
    <div class="delta-wrap">
      {delta_svg}
      <p class="chart-sub delta-caption">Best mark (ink) over the population mean (grey);
      each arrow runs from a skeleton node's mean position to its position in the best
      mark.{scale_note}</p>
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
  </section>
"""


# ---------------------------------------------------------------------------
# Whole document
# ---------------------------------------------------------------------------

def build_html(state: dict, basis: PCABasis, n_active: int, sections: list[str],
              sizes: list[str], by_size: dict, z_max: float) -> str:
    seeds = html.escape(", ".join(basis.seed_files))
    total_votes = sum(len(by_size[s]) for s in sizes)
    nav = ""
    if len(sizes) > 1:
        links = " &middot; ".join(
            f'<a href="#size-{html.escape(s)}">{html.escape(s)} ({len(by_size[s])})</a>'
            for s in sizes)
        nav = f'<p class="sizenav">Jump to: {links}</p>'

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

  table {{ border-collapse:collapse; }}
  thead th {{ font-weight:600; color:#555; font-size:12px; padding:4px 0; text-align:center; }}
  thead th:first-child {{ width:88px; }}
  .rowlabel {{ text-align:left; padding:0 10px 0 2px; vertical-align:middle; white-space:nowrap; }}
  .rowlabel .pc {{ font-weight:700; font-size:15px; }}
  .rowlabel .star {{ color:#1a7f37; font-size:12px; font-weight:600; }}
  .rowlabel .kind {{ font-size:11px; color:#888; }}
  .rowlabel .kind.edge {{ color:#c0392b; }}
  .settle-tag {{ display:inline-block; font-size:10px; padding:1px 6px; border-radius:9px;
                margin-top:2px; }}
  .settle-tag.settled {{ background:#e6f4ea; color:#1a7f37; }}
  .settle-tag.unsettled {{ background:#f4ede3; color:#a06a1f; }}
  .cell {{ vertical-align:middle; text-align:center; border:1px solid #eee; }}
  .cell.star {{ outline:2px solid #1a7f37; outline-offset:-2px; }}
  .mark {{ display:block; width:82px; height:92px; margin:2px auto 0; }}
  .bar {{ height:26px; display:flex; align-items:flex-end; justify-content:center; }}
  .bar i {{ display:block; width:60%; background:#1a7f37; opacity:.55; }}
  .cell.profile {{ background:#f4fbf6; border-left:2px solid #cfe9d8; }}
  .cell.profile .ptag {{ font-size:11px; color:#1a7f37; padding:2px 0 4px; }}
</style></head>
<body>
  <svg width="0" height="0" style="position:absolute" aria-hidden="true">
    <defs>
      <marker id="deltaArrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill="{ARROW_COLOR}"/>
      </marker>
    </defs>
  </svg>
  <h1>Maker&rsquo;s Mark preferences</h1>
  <p class="sub">One Bayesian peaked model per size bucket, fitted to <b>{total_votes}</b>
  logged duels total across {basis.n_seeds} seeds ({basis.n_components} components,
  {n_active} active). Each section below is one size bucket.</p>
  {nav}
  {"".join(sections)}
  <p class="sub" style="margin-top:20px">
    <b>peak</b> = an interior sweet spot (utility falls off on both sides);
    <b class="warn" style="color:#c0392b">edge</b> = preference keeps rising to the
    &plusmn;{z_max:g}&sigma; bound, no interior optimum yet.
    &nbsp;&middot;&nbsp; seeds: {seeds}
  </p>
</body></html>
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
    n_active = state.get("n_active", basis.n_components)
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

    sections = [
        build_size_section(basis, by_size[size], size, n_active, args.z_max,
                           args.steps, args.components)
        for size in sizes_to_render
    ]
    doc = build_html(state, basis, n_active, sections, sizes_to_render, by_size, args.z_max)
    Path(args.out).write_text(doc, encoding="utf-8")

    counts = ", ".join(f"{s}: {len(by_size[s])}" for s in sizes_to_render)
    print(f"rendered {len(sizes_to_render)} size bucket(s) ({counts}); wrote {args.out}")


if __name__ == "__main__":
    main()
