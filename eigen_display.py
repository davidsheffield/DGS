"""Visualize the eigenshape space fitted over ``Samples/vector_*.svg``.

``eigen.py`` learns a PCA basis whose axes are whole-mark deformation
directions.  Numbers in ``state.json`` don't tell you what those axes *mean*,
so this renders them: for each principal component, draw the mark at the mean
and stepped +/- a few standard deviations along that one axis (every other
coefficient held at zero).  Walking a row shows exactly what that eigenvector
does to the drawing; the last column overlays the whole walk so the direction
of motion is visible at a glance.

Pure standard library; writes a single self-contained ``eigenshapes.html``.

    python3 eigen_display.py                 # all PCs, +/-2 std, 5 steps
    python3 eigen_display.py --components 12 --sigma 2.5 --steps 7
    python3 eigen_display.py -o out.html
"""

from __future__ import annotations

import argparse
import html

from genome import PATH_ORDER, STROKE_WIDTH, VIEWBOX
from eigen import PCABasis
from genome import load_samples


def _lerp_color(t: float) -> str:
    """t in [-1, 1] -> hex color: blue (neg) -> grey (mean) -> red (pos)."""
    if t <= 0.0:
        a = min(1.0, -t)                      # 0 at mean, 1 at most-negative
        r, g, b = (1 - a) * 0.45 + a * 0.16, (1 - a) * 0.45 + a * 0.40, (1 - a) * 0.45 + a * 0.85
    else:
        a = min(1.0, t)
        r, g, b = (1 - a) * 0.45 + a * 0.85, (1 - a) * 0.45 + a * 0.16, (1 - a) * 0.45 + a * 0.18
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def _paths_at(basis: PCABasis, comp: int, coeff: float) -> list[str]:
    """The mark's path-``d`` strings at ``mean + coeff*component[comp]``."""
    c = [0.0] * basis.n_components
    c[comp] = coeff
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


def build_html(basis: PCABasis, n_components: int, sigma: float,
               steps: int) -> str:
    total = sum(basis.eigenvalues) or 1.0
    n_comp = min(n_components, basis.n_components)
    if steps < 2:
        steps = 2
    # symmetric multipliers of std across [-sigma, +sigma], mean guaranteed at 0
    if steps % 2 == 0:
        steps += 1
    half = steps // 2
    mults = [(-sigma + 2 * sigma * k / (steps - 1)) for k in range(steps)]
    mults[half] = 0.0                          # exact mean in the center

    col_head = "".join(
        f'<th>{("mean" if abs(m) < 1e-9 else f"{m:+.2g}σ")}</th>'
        for m in mults
    ) + "<th>overlay</th>"

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
        label = (f'<div class="pc">PC{i + 1}</div>'
                 f'<div class="var">{var_pct:.1f}% var</div>'
                 f'<div class="std">σ={std:.1f}</div>')
        rows.append(f'<tr><th class="rowlabel">{label}</th>{"".join(cells)}</tr>')

    top_n = sum(basis.eigenvalues[:n_comp]) / total * 100.0
    seeds = html.escape(", ".join(basis.seed_files))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Maker's Mark eigenshapes</title>
<style>
  :root {{ --ink:#231f20; }}
  body {{ font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif; color:var(--ink);
         margin:0; padding:24px 28px; background:#fafafa; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
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
  tbody tr:nth-child(even) td {{ background:#f2f2f2; }}
  .legend {{ margin-top:18px; color:#666; font-size:12px; }}
  .legend b {{ color:var(--ink); }}
  .swatch {{ display:inline-block; width:12px; height:12px; border-radius:2px;
             vertical-align:-2px; margin:0 3px 0 8px; }}
</style></head>
<body>
  <h1>Maker&rsquo;s Mark eigenshapes</h1>
  <p class="sub">PCA basis fitted over {basis.n_seeds} seed marks
  ({basis.dim} features, {basis.n_components} components). Each row is one
  principal component: the mark decoded at the population <b>mean</b> and
  stepped along that single axis by multiples of its standard deviation
  &sigma; (all other coefficients held at zero). The top {n_comp} components
  shown here explain <b>{top_n:.1f}%</b> of the variance across the seeds.</p>
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
    &nbsp;&middot;&nbsp; seeds: {seeds}
  </p>
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
    args = ap.parse_args()

    pop = load_samples()
    basis = PCABasis.fit(pop)
    n_components = args.components if args.components is not None else basis.n_components
    doc = build_html(basis, n_components, args.sigma, args.steps)
    with open(args.out, "w") as fh:
        fh.write(doc)
    top = min(n_components, basis.n_components)
    print(f"fitted {basis.n_components} components from {basis.n_seeds} seeds; "
          f"wrote top {top} to {args.out}")


if __name__ == "__main__":
    main()
