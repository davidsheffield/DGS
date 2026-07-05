"""Visualize the eigenshape space fitted over ``Samples/vector_*.svg``.

``eigen.py`` learns a PCA basis whose axes are whole-mark deformation
directions.  Numbers in ``state.json`` don't tell you what those axes *mean*,
so this renders them: for each principal component, draw the mark at the mean
and stepped +/- a few standard deviations along that one axis (every other
coefficient held at zero).  Walking a row shows exactly what that eigenvector
does to the drawing; a "motion" column draws it as a displacement field, and
the overlay column superimposes the whole walk so the direction of motion is
visible at a glance.  A scree plot up top shows how many axes carry real
signal, and a two-axis morph section at the bottom shows how pairs of axes
combine (something no single-axis row can show).

Pure standard library; writes a single self-contained ``eigenshapes.html``.

    python3 eigen_display.py                 # all PCs, +/-2 std, 5 steps
    python3 eigen_display.py --components 12 --sigma 2.5 --steps 7
    python3 eigen_display.py --sheet-sigma 2 --sheet-steps 7 --sheet-pairs 1x2,1x3
    python3 eigen_display.py -o out.html
"""

from __future__ import annotations

import argparse
import html
import math
import statistics

from genome import PATH_ORDER, STROKE_WIDTH, VIEWBOX, CANVAS_W, CANVAS_H
from eigen import PCABasis
from genome import load_samples

# Colors shared with the rest of the page's blue(-sigma)/grey(mean)/red(+sigma)
# language (see _lerp_color below).
BLUE = "#2966d9"
GREY = "#737373"
RED = "#d92e2e"
INK = "#231f20"


def _lerp_color(t: float) -> str:
    """t in [-1, 1] -> hex color: blue (neg) -> grey (mean) -> red (pos)."""
    if t <= 0.0:
        a = min(1.0, -t)                      # 0 at mean, 1 at most-negative
        r, g, b = (1 - a) * 0.45 + a * 0.16, (1 - a) * 0.45 + a * 0.40, (1 - a) * 0.45 + a * 0.85
    else:
        a = min(1.0, t)
        r, g, b = (1 - a) * 0.45 + a * 0.85, (1 - a) * 0.45 + a * 0.16, (1 - a) * 0.45 + a * 0.18
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def _mult_label(m: float) -> str:
    return "mean" if abs(m) < 1e-9 else f"{m:+.2g}σ"


def _step_mults(sigma: float, steps: int) -> list[float]:
    """Symmetric multipliers of std across [-sigma, +sigma], mean at 0."""
    if steps < 2:
        steps = 2
    if steps % 2 == 0:
        steps += 1
    half = steps // 2
    mults = [(-sigma + 2 * sigma * k / (steps - 1)) for k in range(steps)]
    mults[half] = 0.0
    return mults


def _paths_at(basis: PCABasis, comp: int, coeff: float) -> list[str]:
    """The mark's path-``d`` strings at ``mean + coeff*component[comp]``."""
    c = [0.0] * basis.n_components
    c[comp] = coeff
    g = basis.decode(c)
    return [g.paths[pid].to_d() for pid in PATH_ORDER]


def _paths_at_2d(basis: PCABasis, comp_x: int, coeff_x: float,
                 comp_y: int, coeff_y: float) -> list[str]:
    c = [0.0] * basis.n_components
    c[comp_x] = coeff_x
    c[comp_y] += coeff_y
    g = basis.decode(c)
    return [g.paths[pid].to_d() for pid in PATH_ORDER]


def _svg_cell(path_ds: list[str], color: str, width: float,
              extra: str = "") -> str:
    body = "".join(
        f'<path d="{d}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-miterlimit="10"/>'
        for d in path_ds
    )
    return (f'<svg viewBox="{VIEWBOX}" preserveAspectRatio="xMidYMid meet" '
            f'class="mark">{extra}{body}</svg>')


# ---------------------------------------------------------------------------
# 1. Scree / cumulative-variance chart
# ---------------------------------------------------------------------------

def _bar_path(x: float, y_base: float, y_top: float, w: float, r: float) -> str:
    """Bar from ``y_top`` to the baseline ``y_base``, rounded top corners."""
    h = y_base - y_top
    if h <= 0.05:
        return ""
    r = max(0.0, min(r, w / 2.0, h))
    return (f"M{x:.2f},{y_base:.2f} L{x:.2f},{y_top + r:.2f} "
            f"Q{x:.2f},{y_top:.2f} {x + r:.2f},{y_top:.2f} "
            f"L{x + w - r:.2f},{y_top:.2f} "
            f"Q{x + w:.2f},{y_top:.2f} {x + w:.2f},{y_top + r:.2f} "
            f"L{x + w:.2f},{y_base:.2f} Z")


def _scree_chart_svg(basis: PCABasis, n_comp: int) -> str:
    total = sum(basis.eigenvalues) or 1.0
    var_pct = [basis.eigenvalues[i] / total * 100.0 for i in range(n_comp)]
    cum_pct = []
    running = 0.0
    for v in var_pct:
        running += v
        cum_pct.append(running)

    left, right, top, bottom = 46, 20, 16, 176
    plot_w = 26 * n_comp
    width = left + plot_w + right
    height = 214

    def y_of(pct: float) -> float:
        return bottom - (pct / 100.0) * (bottom - top)

    band = plot_w / n_comp
    bar_w = min(22.0, band * 0.62)

    # gridlines + y ticks at 0/25/50/75/100
    grid = []
    for tick in (0, 25, 50, 75, 100):
        y = y_of(tick)
        grid.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w:.2f}" '
                    f'y2="{y:.2f}" stroke="#ddd" stroke-width="1"/>')
        grid.append(f'<text x="{left - 8}" y="{y + 3.5:.2f}" text-anchor="end" '
                    f'class="axislabel">{tick}%</text>')
    grid.append(f'<line x1="{left}" y1="{bottom}" x2="{left + plot_w:.2f}" '
                f'y2="{bottom}" stroke="#bbb" stroke-width="1"/>')

    # bars: per-component variance share
    stride = max(1, -(-n_comp // 15))          # show <=~15 x-axis labels
    bars = []
    xticks = []
    for i in range(n_comp):
        x = left + i * band + (band - bar_w) / 2.0
        y_top = y_of(var_pct[i])
        d = _bar_path(x, bottom, y_top, bar_w, 3.5)
        title = (f"PC{i + 1}: {var_pct[i]:.1f}% of variance "
                 f"(cumulative {cum_pct[i]:.1f}%), σ={basis.stds[i]:.2f}")
        if d:
            bars.append(f'<path d="{d}" fill="{BLUE}"><title>{html.escape(title)}'
                        f'</title></path>')
        else:
            bars.append(f'<rect x="{x:.2f}" y="{bottom - 0.5}" width="{bar_w:.2f}" '
                        f'height="0.5" fill="{BLUE}"><title>{html.escape(title)}'
                        f'</title></rect>')
        if i == 0 or i == n_comp - 1 or i % stride == 0:
            cx = left + i * band + band / 2.0
            xticks.append(f'<text x="{cx:.2f}" y="{bottom + 14}" text-anchor="middle" '
                          f'class="axislabel">PC{i + 1}</text>')

    # cumulative line + markers
    pts = [(left + i * band + band / 2.0, y_of(cum_pct[i])) for i in range(n_comp)]
    line_d = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in pts)
    line = f'<path d="{line_d}" fill="none" stroke="{INK}" stroke-width="2" stroke-linejoin="round"/>'
    markers = []
    for i, (x, y) in enumerate(pts):
        title = (f"PC{i + 1}: cumulative {cum_pct[i]:.1f}% "
                 f"(+{var_pct[i]:.1f}%)")
        markers.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{INK}" '
                       f'stroke="#fff" stroke-width="2"><title>'
                       f'{html.escape(title)}</title></circle>')
    # endpoint label on the cumulative line (the extreme worth calling out)
    last_x, last_y = pts[-1]
    end_label = (f'<text x="{last_x:.2f}" y="{last_y - 10:.2f}" text-anchor="end" '
                f'class="endlabel">{cum_pct[-1]:.1f}%</text>')

    body = "".join(grid) + "".join(bars) + "".join(xticks) + line + "".join(markers) + end_label
    return (f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'class="screesvg">{body}</svg>')


# ---------------------------------------------------------------------------
# 2. Displacement-field ("motion") column
# ---------------------------------------------------------------------------

def _motion_amp(basis: PCABasis, n_comp: int, sigma: float) -> float:
    """Common amplification factor so a *typical* node displacement reads.

    One factor for the whole page (not per-row) so arrow length stays
    comparable across components -- a genuinely more active axis still shows
    longer arrows than a settled one.
    """
    mean_g = basis.decode([0.0] * basis.n_components)
    mean_nodes = {pid: mean_g.paths[pid].nodes for pid in PATH_ORDER}
    mags = []
    for i in range(n_comp):
        c = [0.0] * basis.n_components
        c[i] = sigma * basis.stds[i]
        g = basis.decode(c)
        for pid in PATH_ORDER:
            for (mx, my), (dx, dy) in zip(mean_nodes[pid], g.paths[pid].nodes):
                mags.append(math.hypot(dx - mx, dy - my))
    med = statistics.median(mags) if mags else 1.0
    target = 7.0
    amp = target / med if med > 1e-9 else 1.0
    nice_steps = [1, 1.5, 2, 3, 4, 5, 8, 10, 15, 20, 30, 50]
    return min(nice_steps, key=lambda x: abs(x - amp)) if amp > 1.0 else 1.0


def _arrow_svg(x0: float, y0: float, x1: float, y1: float, color: str) -> str:
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 0.12:
        return ""
    ang = math.atan2(dy, dx)
    head = 2.2
    a1, a2 = ang + math.radians(150), ang - math.radians(150)
    hx1, hy1 = x1 + head * math.cos(a1), y1 + head * math.sin(a1)
    hx2, hy2 = x1 + head * math.cos(a2), y1 + head * math.sin(a2)
    return (f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
            f'stroke="{color}" stroke-width="0.7" stroke-linecap="round"/>'
            f'<path d="M{x1:.2f},{y1:.2f} L{hx1:.2f},{hy1:.2f} L{hx2:.2f},{hy2:.2f} Z" '
            f'fill="{color}"/>')


def _clamp_point(mx: float, my: float, dx: float, dy: float,
                 margin: float = 12.0) -> tuple[float, float]:
    """Cap an (already amplified) displaced point so arrows can't blow past
    the card -- direction is preserved, only the length is capped."""
    lo_x, hi_x = -margin, CANVAS_W + margin
    lo_y, hi_y = -margin, CANVAS_H + margin
    tx, ty = mx + dx, my + dy
    if lo_x <= tx <= hi_x and lo_y <= ty <= hi_y:
        return tx, ty
    scale = 1.0
    if dx > 0 and tx > hi_x:
        scale = min(scale, (hi_x - mx) / dx)
    elif dx < 0 and tx < lo_x:
        scale = min(scale, (lo_x - mx) / dx)
    if dy > 0 and ty > hi_y:
        scale = min(scale, (hi_y - my) / dy)
    elif dy < 0 and ty < lo_y:
        scale = min(scale, (lo_y - my) / dy)
    scale = max(0.0, scale)
    return mx + dx * scale, my + dy * scale


def _motion_svg(basis: PCABasis, comp: int, sigma: float, amp: float) -> str:
    mean_g = basis.decode([0.0] * basis.n_components)
    c = [0.0] * basis.n_components
    c[comp] = sigma * basis.stds[comp]
    moved_g = basis.decode(c)

    mean_layer = "".join(
        f'<path d="{mean_g.paths[pid].to_d()}" fill="none" stroke="{GREY}" '
        f'stroke-width="{STROKE_WIDTH * 0.6}" stroke-miterlimit="10" opacity="0.55"/>'
        for pid in PATH_ORDER
    )
    dots, arrows = [], []
    for pid in PATH_ORDER:
        for (mx, my), (tx, ty) in zip(mean_g.paths[pid].nodes, moved_g.paths[pid].nodes):
            dx, dy = (tx - mx) * amp, (ty - my) * amp
            ex, ey = _clamp_point(mx, my, dx, dy)
            dots.append(f'<circle cx="{mx:.2f}" cy="{my:.2f}" r="1.1" fill="{GREY}"/>')
            arrows.append(_arrow_svg(mx, my, ex, ey, RED))
    body = mean_layer + "".join(dots) + "".join(arrows)
    return (f'<svg viewBox="{VIEWBOX}" preserveAspectRatio="xMidYMid meet" '
            f'class="mark">{body}</svg>')


# ---------------------------------------------------------------------------
# 3. Two-axis morph sheets
# ---------------------------------------------------------------------------

def _default_sheet_pairs(n_components: int) -> list[tuple[int, int]]:
    candidates = [(1, 2), (1, 3), (2, 3)]
    return [(a, b) for a, b in candidates if a <= n_components and b <= n_components]


def _parse_sheet_pairs(spec: str, n_components: int) -> list[tuple[int, int]]:
    pairs = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        a_s, _, b_s = tok.lower().partition("x")
        a, b = int(a_s), int(b_s)
        if a < 1 or b < 1 or a > n_components or b > n_components:
            raise ValueError(f"--sheet-pairs entry {tok!r} is out of range "
                             f"(basis has {n_components} components)")
        pairs.append((a, b))
    return pairs


def _sheet_section(basis: PCABasis, a: int, b: int, sheet_sigma: float,
                   sheet_steps: int) -> str:
    """One PC-a (x) x PC-b (y) morph grid, PC indices 1-based."""
    ix, iy = a - 1, b - 1
    x_mults = _step_mults(sheet_sigma, sheet_steps)      # left -> right
    y_mults = list(reversed(_step_mults(sheet_sigma, sheet_steps)))  # top -> bottom

    col_head = "".join(f'<th>{_mult_label(m)}</th>' for m in x_mults)
    rows = []
    for my in y_mults:
        cy = my * basis.stds[iy]
        cells = []
        for mx in x_mults:
            cx = mx * basis.stds[ix]
            ds = _paths_at_2d(basis, ix, cx, iy, cy)
            is_mean = abs(mx) < 1e-9 and abs(my) < 1e-9
            cls = "cell mean" if is_mean else "cell"
            cells.append(f'<td class="{cls}">{_svg_cell(ds, INK, STROKE_WIDTH)}</td>')
        rows.append(f'<tr><th class="rowlabel sheet">{_mult_label(my)}</th>{"".join(cells)}</tr>')

    var_a = basis.eigenvalues[ix] / (sum(basis.eigenvalues) or 1.0) * 100.0
    var_b = basis.eigenvalues[iy] / (sum(basis.eigenvalues) or 1.0) * 100.0
    return f"""
  <div class="sheet-wrap">
    <h3>PC{a} ({var_a:.1f}% var) &times; PC{b} ({var_b:.1f}% var)</h3>
    <div class="scrollx">
    <table class="sheet">
      <thead><tr><th></th>{col_head}</tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    </div>
  </div>"""


def build_html(basis: PCABasis, n_components: int, sigma: float, steps: int,
              sheet_sigma: float = 1.5, sheet_steps: int = 5,
              sheet_pairs: list[tuple[int, int]] | None = None) -> str:
    total = sum(basis.eigenvalues) or 1.0
    n_comp = min(n_components, basis.n_components)
    mults = _step_mults(sigma, steps)

    amp = _motion_amp(basis, n_comp, sigma)

    col_head = "".join(f'<th>{_mult_label(m)}</th>' for m in mults) + "<th>overlay</th><th>motion</th>"

    rows = []
    for i in range(n_comp):
        std = basis.stds[i]
        var_pct = basis.eigenvalues[i] / total * 100.0
        cells = []
        overlay_layers = []
        for m in mults:
            t = 0.0 if sigma == 0 else m / sigma
            color = "#231f20" if abs(m) < 1e-9 else _lerp_color(t)
            ds = _paths_at(basis, i, m * std)
            is_mean = abs(m) < 1e-9
            cls = "cell mean" if is_mean else "cell"
            cells.append(f'<td class="{cls}">{_svg_cell(ds, color, STROKE_WIDTH)}</td>')
            w = STROKE_WIDTH if is_mean else STROKE_WIDTH * 0.7
            overlay_layers.append("".join(
                f'<path d="{d}" fill="none" stroke="{color}" '
                f'stroke-width="{w}" stroke-miterlimit="10" '
                f'opacity="{1.0 if is_mean else 0.75}"/>'
                for d in ds
            ))
        overlay = (f'<svg viewBox="{VIEWBOX}" preserveAspectRatio="xMidYMid meet" '
                   f'class="mark">{"".join(overlay_layers)}</svg>')
        cells.append(f'<td class="cell overlay">{overlay}</td>')
        cells.append(f'<td class="cell motion">{_motion_svg(basis, i, sigma, amp)}</td>')
        label = (f'<div class="pc">PC{i + 1}</div>'
                 f'<div class="var">{var_pct:.1f}% var</div>'
                 f'<div class="std">σ={std:.1f}</div>')
        rows.append(f'<tr><th class="rowlabel">{label}</th>{"".join(cells)}</tr>')

    top_n = sum(basis.eigenvalues[:n_comp]) / total * 100.0
    seeds = html.escape(", ".join(basis.seed_files))

    scree_svg = _scree_chart_svg(basis, n_comp)

    if sheet_pairs is None:
        sheet_pairs = _default_sheet_pairs(basis.n_components)
    sheet_sections = "".join(
        _sheet_section(basis, a, b, sheet_sigma, sheet_steps) for a, b in sheet_pairs
    )
    sheets_block = ""
    if sheet_sections:
        sheets_block = f"""
  <h2>Two-axis morph sheets</h2>
  <p class="sub">Every row above holds every axis but one at the mean; that
  can't show how two axes <em>combine</em>. Each grid below decodes the mark
  at ({sheet_steps} &times; {sheet_steps}) combinations of two components, z
  spaced evenly over &plusmn;{sheet_sigma:g}&sigma;. Columns are the first
  (x) component, rows the second (y, top = +&sigma;); the center cell (outlined
  like the mean cells above) is the population mean.</p>
  {sheet_sections}"""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Maker's Mark eigenshapes</title>
<style>
  :root {{ --ink:#231f20; }}
  body {{ font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif; color:var(--ink);
         margin:0; padding:24px 28px; background:#fafafa; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  h2 {{ font-size:16px; margin:28px 0 4px; }}
  h3 {{ font-size:13px; margin:18px 0 6px; color:#444; }}
  .sub {{ color:#666; margin:0 0 18px; max-width:70ch; }}
  table {{ border-collapse:collapse; }}
  th, td {{ padding:0; }}
  thead th {{ font-weight:600; color:#555; font-size:12px; padding:4px 0; text-align:center; }}
  thead th:first-child {{ width:96px; }}
  .rowlabel {{ text-align:left; padding:0 10px 0 2px; vertical-align:middle; white-space:nowrap; }}
  .rowlabel .pc {{ font-weight:700; font-size:15px; }}
  .rowlabel .var {{ color:#c0392b; font-size:12px; }}
  .rowlabel .std {{ color:#888; font-size:11px; }}
  .cell {{ vertical-align:middle; }}
  .mark {{ display:block; width:104px; height:118px; margin:2px; background:#fff;
           border:1px solid #eee; border-radius:4px; }}
  .cell.mean .mark {{ border-color:#bbb; box-shadow:0 0 0 1px #bbb inset; }}
  .cell.overlay .mark {{ background:#fff; border-color:#ddd; width:120px; height:136px; }}
  .cell.motion .mark {{ background:#fff; border-color:#ddd; width:120px; height:136px; }}
  tbody tr:nth-child(even) td {{ background:#f2f2f2; }}
  .legend {{ margin-top:18px; color:#666; font-size:12px; }}
  .legend b {{ color:var(--ink); }}
  .swatch {{ display:inline-block; width:12px; height:12px; border-radius:2px;
             vertical-align:-2px; margin:0 3px 0 8px; }}
  .scree-wrap {{ background:#fff; border:1px solid #eee; border-radius:4px;
                display:inline-block; padding:8px 10px 4px; }}
  .screesvg {{ display:block; }}
  .axislabel {{ font-size:9px; fill:#777; }}
  .endlabel {{ font-size:11px; font-weight:600; fill:var(--ink); }}
  .scree-legend {{ margin:6px 0 4px; color:#666; font-size:12px; }}
  .rowlabel.sheet {{ font-size:11px; color:#555; text-align:right; padding-right:8px; width:auto; }}
  table.sheet thead th {{ font-size:11px; }}
  .sheet-wrap {{ margin-bottom:8px; }}
  .scrollx {{ overflow-x:auto; }}
</style></head>
<body>
  <h1>Maker&rsquo;s Mark eigenshapes</h1>
  <p class="sub">PCA basis fitted over {basis.n_seeds} seed marks
  ({basis.dim} features, {basis.n_components} components). Each row is one
  principal component: the mark decoded at the population <b>mean</b> and
  stepped along that single axis by multiples of its standard deviation
  &sigma; (all other coefficients held at zero). The top {n_comp} components
  shown here explain <b>{top_n:.1f}%</b> of the variance across the seeds.</p>

  <h2>Scree / cumulative variance</h2>
  <div class="scree-wrap">{scree_svg}</div>
  <p class="scree-legend">
    <span class="swatch" style="background:{BLUE}"></span>per-component variance share
    <span class="swatch" style="background:{INK}"></span>cumulative variance
    &nbsp;&middot;&nbsp; both read off the same 0&ndash;100% axis. Hover a bar or
    dot for its exact figures. Where the line flattens, later components add
    almost nothing &mdash; that's roughly how many axes carry real signal
    versus noise.
  </p>

  <table>
    <thead><tr><th></th>{col_head}</tr></thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
  <p class="legend">
    <span class="swatch" style="background:#2966d9"></span>&minus;&sigma; direction
    <span class="swatch" style="background:#737373"></span>mean
    <span class="swatch" style="background:#d92e2e"></span>+&sigma; direction
    &nbsp;&middot;&nbsp; <b>overlay</b> column superimposes the whole walk.
    &nbsp;&middot;&nbsp; <b>motion</b> column: grey mean mark, red arrows show
    where each on-curve node moves stepping to +&sigma; on that axis (arrows
    &times;{amp:g} so small moves stay visible; grey dots mark each node's
    mean position).
    &nbsp;&middot;&nbsp; seeds: {seeds}
  </p>
  {sheets_block}
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="eigenshapes.html",
                    help="output HTML file (default eigenshapes.html)")
    ap.add_argument("-n", "--components", type=int, default=None,
                    help="number of top principal components to show (default: all)")
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="how many standard deviations to walk each side")
    ap.add_argument("--steps", type=int, default=5,
                    help="cells per row across the walk (odd; mean centered)")
    ap.add_argument("--sheet-sigma", type=float, default=1.5,
                    help="std-devs each way for the two-axis morph sheets (default 1.5)")
    ap.add_argument("--sheet-steps", type=int, default=5,
                    help="grid size per side for the morph sheets (odd; mean centered, default 5)")
    ap.add_argument("--sheet-pairs", default=None,
                    help="comma-separated PC pairs for the morph sheets, e.g. "
                         "'1x2,1x3' (default: PC1x2, PC1x3, PC2x3, as many as exist)")
    args = ap.parse_args()

    pop = load_samples()
    basis = PCABasis.fit(pop)
    n_components = args.components if args.components is not None else basis.n_components
    sheet_pairs = None
    if args.sheet_pairs is not None:
        sheet_pairs = _parse_sheet_pairs(args.sheet_pairs, basis.n_components)
    doc = build_html(basis, n_components, args.sigma, args.steps,
                     sheet_sigma=args.sheet_sigma, sheet_steps=args.sheet_steps,
                     sheet_pairs=sheet_pairs)
    with open(args.out, "w") as fh:
        fh.write(doc)
    top = min(n_components, basis.n_components)
    print(f"fitted {basis.n_components} components from {basis.n_seeds} seeds; "
          f"wrote top {top} to {args.out}")


if __name__ == "__main__":
    main()
