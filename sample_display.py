"""Compare each seed sample to the population mean, decomposed by eigen-axis.

``eigen_display.py`` shows what a principal component *means* by walking it in
isolation from the mean. This is the complementary view: for each sample in
``Samples/vector_*.svg``, how far it actually sits from the mean and which
eigen-axes carry that deviation. ``PCABasis.encode()`` gives an *exact*
per-axis decomposition of a sample's squared distance from the mean -- the
components are orthonormal, so ``sum(c_k**2) == that squared distance``
(Parseval) -- so each cell's z-score and %-of-deviation are not approximations.

Pure standard library; writes a single self-contained ``sample_eigen.html``.

    python3 sample_display.py                    # all components, sorted by distance
    python3 sample_display.py -n 12 --sort file
    python3 sample_display.py -o out.html
"""

from __future__ import annotations

import argparse
import html
import math

from genome import PATH_ORDER, STROKE_WIDTH, VIEWBOX, load_samples
from eigen import PCABasis
from eigen_display import _lerp_color

MEAN_STROKE = "#bbbbbb"
SAMPLE_STROKE = "#231f20"


def _text_color(hexcolor: str) -> str:
    """Ink or near-white, whichever contrasts with a heatmap cell's fill."""
    r, g, b = (int(hexcolor[i:i + 2], 16) for i in (1, 3, 5))
    yiq = (r * 299 + g * 587 + b * 114) / 1000.0
    return "#231f20" if yiq >= 140 else "#fbfbfb"


def _mark_pair_svg(mean_ds: list[str], sample_ds: list[str]) -> str:
    """Mean (grey, behind) overlaid with one sample (ink, on top)."""
    mean_layer = "".join(
        f'<path d="{d}" fill="none" stroke="{MEAN_STROKE}" '
        f'stroke-width="{STROKE_WIDTH}" stroke-miterlimit="10"/>' for d in mean_ds)
    sample_layer = "".join(
        f'<path d="{d}" fill="none" stroke="{SAMPLE_STROKE}" '
        f'stroke-width="{STROKE_WIDTH}" stroke-miterlimit="10"/>' for d in sample_ds)
    return (f'<svg viewBox="{VIEWBOX}" preserveAspectRatio="xMidYMid meet" '
            f'class="mark">{mean_layer}{sample_layer}</svg>')


def build_html(basis: PCABasis, pop: list, n_components: int, sort_mode: str) -> str:
    total_var = sum(basis.eigenvalues) or 1.0
    n_comp = min(n_components, basis.n_components)

    names = [g.meta.get("origin", "?") for g in pop]
    coeffs_list = [basis.encode(g) for g in pop]
    dist2_list = [sum(c * c for c in coeffs) for coeffs in coeffs_list]

    order = list(range(len(pop)))
    if sort_mode == "distance":
        order.sort(key=lambda i: -dist2_list[i])

    zmatrix = [[(coeffs_list[i][k] / basis.stds[k]) if basis.stds[k] else 0.0
               for k in range(n_comp)] for i in range(len(pop))]
    zmax = max([3.0] + [abs(z) for row in zmatrix for z in row])

    mean_g = basis.decode([0.0] * basis.n_components)
    mean_ds = [mean_g.paths[pid].to_d() for pid in PATH_ORDER]

    col_head = "".join(
        f'<th><div class="pc">PC{k + 1}</div>'
        f'<div class="var">{basis.eigenvalues[k] / total_var * 100:.1f}%</div></th>'
        for k in range(n_comp)
    )

    rows = []
    for i in order:
        g = pop[i]
        coeffs = coeffs_list[i]
        dist2 = dist2_list[i]
        dist = math.sqrt(dist2)
        sample_ds = [g.paths[pid].to_d() for pid in PATH_ORDER]
        thumb = _mark_pair_svg(mean_ds, sample_ds)

        contributions = [(k, (coeffs[k] ** 2 / dist2 * 100.0) if dist2 else 0.0)
                         for k in range(basis.n_components)]
        top = sorted(contributions, key=lambda kv: -kv[1])[:3]
        top_text = ", ".join(f"PC{k + 1} {pct:.0f}%" for k, pct in top if pct >= 1.0) or "—"

        cells = []
        for k in range(n_comp):
            z = zmatrix[i][k]
            fill = _lerp_color(max(-1.0, min(1.0, z / zmax)))
            txt = _text_color(fill)
            pct = (coeffs[k] ** 2 / dist2 * 100.0) if dist2 else 0.0
            title = (f"PC{k + 1}: z={z:+.2f}σ, coeff={coeffs[k]:+.3f}, "
                    f"{pct:.1f}% of this sample's deviation from the mean")
            cells.append(
                f'<td class="cell" style="background:{fill}" title="{html.escape(title)}">'
                f'<span style="color:{txt}">{z:+.1f}</span></td>'
            )

        label = (f'{thumb}<div class="name">{html.escape(names[i])}</div>'
                 f'<div class="dist">Δ={dist:.2f}</div>'
                 f'<div class="top">{top_text}</div>')
        rows.append(f'<tr><th class="rowlabel">{label}</th>{"".join(cells)}</tr>')

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Maker's Mark samples vs. mean</title>
<style>
  :root {{ --ink:#231f20; }}
  body {{ font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif; color:var(--ink);
         margin:0; padding:24px 28px; background:#fafafa; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  .sub {{ color:#666; margin:0 0 18px; max-width:76ch; }}
  table {{ border-collapse:collapse; }}
  th, td {{ padding:0; }}
  thead th {{ font-weight:600; color:#555; font-size:11px; padding:4px 2px; text-align:center;
             vertical-align:bottom; }}
  thead th .pc {{ font-weight:700; font-size:12px; }}
  thead th .var {{ color:#c0392b; font-size:10px; }}
  .rowlabel {{ text-align:left; padding:4px 10px 4px 2px; vertical-align:middle; white-space:nowrap; }}
  .rowlabel .name {{ font-weight:700; font-size:13px; }}
  .rowlabel .dist {{ color:#888; font-size:11px; }}
  .rowlabel .top {{ color:#555; font-size:11px; }}
  .mark {{ display:block; width:72px; height:82px; margin:2px 0; background:#fff;
           border:1px solid #eee; border-radius:4px; }}
  .cell {{ width:44px; height:44px; text-align:center; vertical-align:middle;
           font-size:11px; font-variant-numeric:tabular-nums; cursor:default; }}
  tbody tr:nth-child(even) th.rowlabel {{ background:#f2f2f2; }}
  .legend {{ margin-top:18px; color:#666; font-size:12px; max-width:76ch; }}
  .legend b {{ color:var(--ink); }}
  .swatch {{ display:inline-block; width:12px; height:12px; border-radius:2px;
             vertical-align:-2px; margin:0 3px 0 8px; }}
</style></head>
<body>
  <h1>Maker&rsquo;s Mark samples vs. mean</h1>
  <p class="sub">PCA basis fitted over {basis.n_seeds} seed marks ({basis.dim} features,
  {basis.n_components} components). Each row is one sample from <code>Samples/</code>:
  its mark overlaid on the population <b>mean</b> (grey), how far it sits from the mean
  in feature space (&Delta;, the L2 distance whose square splits exactly across the
  columns), and the eigen-axes ({("PC" + str(n_comp)) if n_comp < basis.n_components else "all " + str(n_comp) + " components"}
  shown) that carry the most of that deviation. Each heatmap cell is that sample's
  coefficient on one principal component, in standard deviations (&sigma;) of how much
  the seed population itself varies along that axis &mdash; hover a cell for the exact
  coefficient and its share of the sample's total deviation. Column headers show what
  share of the population's total variance that axis explains. Rows are sorted by
  {"distance from the mean, largest first" if sort_mode == "distance" else "sample file order"}.</p>
  <table>
    <thead><tr><th></th>{col_head}</tr></thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
  <p class="legend">
    <span class="swatch" style="background:#2966d9"></span>below the axis's mean (&minus;&sigma;)
    <span class="swatch" style="background:#737373"></span>at the mean
    <span class="swatch" style="background:#d92e2e"></span>above the axis's mean (+&sigma;)
    &nbsp;&middot;&nbsp; grey mark = population mean, ink mark = this sample
    &nbsp;&middot;&nbsp; color scale saturates at &plusmn;{zmax:.1f}&sigma;
  </p>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="sample_eigen.html",
                    help="output HTML file (default sample_eigen.html)")
    ap.add_argument("-n", "--components", type=int, default=None,
                    help="number of top principal components to show (default: all)")
    ap.add_argument("--sort", choices=("distance", "file"), default="distance",
                    help="row order: farthest-from-mean first, or Samples/ file order")
    args = ap.parse_args()

    pop = load_samples()
    basis = PCABasis.fit(pop)
    n_components = args.components if args.components is not None else basis.n_components
    doc = build_html(basis, pop, n_components, args.sort)
    with open(args.out, "w") as fh:
        fh.write(doc)
    top = min(n_components, basis.n_components)
    print(f"fitted {basis.n_components} components from {basis.n_seeds} seeds; "
          f"compared {len(pop)} samples across top {top} to {args.out}")


if __name__ == "__main__":
    main()
