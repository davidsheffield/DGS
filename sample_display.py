"""Compare each seed sample to the population mean, decomposed by eigen-axis.

``eigen_display.py`` shows what a principal component *means* by walking it in
isolation from the mean. This is the complementary view: for each sample in
``Samples/vector_*.svg``, how far it actually sits from the mean and which
eigen-axes carry that deviation. ``PCABasis.encode()`` gives an *exact*
per-axis decomposition of a sample's squared distance from the mean -- the
components are orthonormal, so ``sum(c_k**2) == that squared distance``
(Parseval) -- so each cell's z-score and %-of-deviation are not approximations.

The page is interactive: a panel to the right of the table (sticky for the
table's height, so it doesn't follow you down into the eigenspace plots below)
shows a larger view. Click a sample's thumbnail (or name) for a mean-vs-sample
comparison; click one of its heatmap cells to keep the full sample and show
it with *that* component's contribution subtracted back out, so you can see
what that one axis was doing to the sample.

A second section plots every sample by two chosen principal-component
coefficients, k-means clusters that 2D projection at a user-selected cluster
count, and shows a knee plot (inertia vs. cluster count) for the same axis
pair so you can judge how many groups the data actually supports.

Pure standard library; writes a single self-contained ``sample_eigen.html``.

    python3 sample_display.py                    # all components, sorted by distance
    python3 sample_display.py -n 12 --sort file
    python3 sample_display.py --max-k 10
    python3 sample_display.py -o out.html
"""

from __future__ import annotations

import argparse
import html
import json
import math

from genome import (CANVAS_H, CANVAS_W, MIN_HANDLE, PATH_ORDER, SEGMENTS,
                    STROKE_WIDTH, VIEWBOX, load_samples)
from eigen import PCABasis
from eigen_display import _lerp_color

MEAN_STROKE = "#bbbbbb"
SAMPLE_STROKE = "#231f20"

# Validated categorical palette (dataviz skill, references/palette.md), fixed order --
# used only for cluster identity in the eigenspace scatter below.
CLUSTER_COLORS = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]
KNEE_COLOR = "#2a78d6"


def _round(seq, nd=6):
    return [round(float(v), nd) for v in seq]


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


def build_html(basis: PCABasis, pop: list, n_components: int, sort_mode: str,
               max_k: int) -> str:
    total_var = sum(basis.eigenvalues) or 1.0
    n_comp = min(n_components, basis.n_components)
    max_k = max(1, min(max_k, len(pop)))

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
    rows_data = []
    for pos, i in enumerate(order):
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
        comps_data = []
        for k in range(n_comp):
            z = zmatrix[i][k]
            fill = _lerp_color(max(-1.0, min(1.0, z / zmax)))
            txt = _text_color(fill)
            pct = (coeffs[k] ** 2 / dist2 * 100.0) if dist2 else 0.0
            title = (f"PC{k + 1}: z={z:+.2f}σ, coeff={coeffs[k]:+.3f}, "
                    f"{pct:.1f}% of this sample's deviation from the mean")
            cells.append(
                f'<td class="cell" data-row="{pos}" data-col="{k}" '
                f'style="background:{fill}" title="{html.escape(title)}">'
                f'<span style="color:{txt}">{z:+.1f}</span></td>'
            )
            residual_coeffs = list(coeffs)
            residual_coeffs[k] = 0.0
            residual_g = basis.decode(residual_coeffs)
            residual_ds = [residual_g.paths[pid].to_d() for pid in PATH_ORDER]
            comps_data.append({"d": residual_ds, "z": z, "coeff": coeffs[k],
                               "pct": pct, "color": fill})

        label = (f'{thumb}<div class="name">{html.escape(names[i])}</div>'
                 f'<div class="dist">Δ={dist:.2f}</div>'
                 f'<div class="top">{top_text}</div>')
        rows.append(
            f'<tr><th class="rowlabel clickable" data-row="{pos}">{label}</th>'
            f'{"".join(cells)}</tr>'
        )
        rows_data.append({"name": names[i], "dist": dist, "full": sample_ds,
                          "comps": comps_data, "coeffsFull": _round(coeffs)})

    var_pct = [basis.eigenvalues[k] / total_var * 100.0 for k in range(n_comp)]
    viz_data = {
        "meanDs": mean_ds, "nComp": n_comp, "varPct": var_pct, "rows": rows_data,
        # Full basis, so cluster centroids (averaged across *all* components, not
        # just the shown ones) can be decoded to a mark client-side, the same way
        # eigen_explorer.py's slider preview does.
        "meanFeat": _round(basis.mean), "weights": _round(basis.weights),
        "components": [_round(basis.components[k]) for k in range(basis.n_components)],
        "canvasW": CANVAS_W, "canvasH": CANVAS_H, "minHandle": MIN_HANDLE,
        "pathOrder": list(PATH_ORDER), "segments": {pid: SEGMENTS[pid] for pid in PATH_ORDER},
    }
    data_json = json.dumps(viz_data, separators=(",", ":")).replace("</", "<\\/")
    cluster_colors_json = json.dumps(CLUSTER_COLORS)

    if n_comp >= 2:
        cluster_section = f"""
  <hr class="section-divider">
  <h2>Samples in eigenspace</h2>
  <p class="sub">Each point is one sample, plotted by two chosen principal-component
  coefficients &mdash; the same coordinates <code>PCABasis.decode()</code> would use to
  reconstruct it. K-means clusters the currently plotted 2D coefficients (colors below);
  the knee plot shows inertia (within-cluster sum of squared distances) across cluster
  counts for that same axis pair, to help judge how many groups the projection actually
  supports. Click a point to jump to that sample's comparison panel near the
  top of the page.</p>
  <div class="cluster-controls">
    <label>X axis <select id="axisX"></select></label>
    <label>Y axis <select id="axisY"></select></label>
    <label>Clusters (k=<span id="kLabel">3</span>)
      <input type="range" id="kSlider" min="1" max="{max_k}" value="3"></label>
  </div>
  <div class="cluster-charts">
    <div class="chart-box">
      <h3>Scatter</h3>
      <p class="chart-sub" id="scatterSub"></p>
      <div id="scatter"></div>
      <div class="legend-row" id="scatterLegend"></div>
    </div>
    <div class="chart-box">
      <h3>Knee plot</h3>
      <p class="chart-sub">Inertia vs. cluster count, this axis pair</p>
      <div id="knee"></div>
    </div>
    <div class="chart-box centroids-box">
      <div class="centroid-head">
        <h3>Cluster centroids</h3>
        <button type="button" id="btnCentroidMean" class="mean-toggle on">mean</button>
      </div>
      <p class="chart-sub">Each cluster's centroid, decoded back to a mark &mdash; the
      average coefficient vector of its members across all {basis.n_components}
      components (not just the two plotted above), so it reflects the whole shape,
      not just this axis pair. Shown separately, then all superimposed. Toggle the
      grey population-mean backdrop with the button above.</p>
      <div class="centroid-row" id="centroidRow"></div>
      <div class="centroid-combined">
        <div class="centroid-overlay" id="centroidOverlay"></div>
        <div class="chart-sub" id="centroidOverlaySub">all clusters overlaid on the mean (grey)</div>
      </div>
    </div>
  </div>
"""
    else:
        cluster_section = """
  <hr class="section-divider">
  <h2>Samples in eigenspace</h2>
  <p class="sub">Needs at least 2 components to plot -- re-run with a larger -n.</p>
"""

    script = _SCRIPT_TEMPLATE
    script = script.replace("__DATA__", data_json)
    script = script.replace("__CLUSTER_COLORS__", cluster_colors_json)
    script = script.replace("__MAX_K__", str(max_k))
    script = script.replace("__STROKE_WIDTH__", json.dumps(STROKE_WIDTH))
    script = script.replace("__VIEWBOX__", VIEWBOX)
    script = script.replace("__N_COMP__", str(n_comp))
    script = script.replace("__HAS_CLUSTERS__", "true" if n_comp >= 2 else "false")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Maker's Mark samples vs. mean</title>
<style>
  :root {{ --ink:#231f20; --knee:{KNEE_COLOR}; }}
  body {{ font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif; color:var(--ink);
         margin:0; padding:24px 28px; background:#fafafa; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  h2 {{ font-size:17px; margin:0 0 4px; }}
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
           font-size:11px; font-variant-numeric:tabular-nums; }}
  .rowlabel.clickable, td.cell {{ cursor:pointer; }}
  .rowlabel.clickable:hover, td.cell:hover {{ outline:2px solid #999; outline-offset:-2px; }}
  tbody tr:nth-child(even) th.rowlabel {{ background:#f2f2f2; }}
  .legend {{ margin-top:18px; color:#666; font-size:12px; max-width:76ch; }}
  .legend b {{ color:var(--ink); }}
  .swatch {{ display:inline-block; width:12px; height:12px; border-radius:2px;
             vertical-align:-2px; margin:0 3px 0 8px; }}
  .layout {{ display:flex; align-items:flex-start; gap:28px; }}
  .table-col {{ flex:1 1 auto; min-width:0; }}
  .big-col {{ flex:0 0 260px; position:sticky; top:16px; align-self:flex-start; }}
  .bigpanel {{ background:#fff; border:1px solid #eee; border-radius:8px;
              padding:14px; display:flex; flex-direction:column; gap:12px; }}
  .bigpanel .mark {{ width:220px; height:250px; }}
  .bigpanel .bigcaption {{ color:#444; font-size:13px; }}
  .bigpanel .bigcaption b {{ color:var(--ink); }}
  .section-divider {{ margin:36px 0 18px; border:none; border-top:1px solid #ddd; }}
  .cluster-controls {{ display:flex; flex-wrap:wrap; gap:20px; align-items:center;
                       margin-bottom:16px; font-size:13px; color:#444; }}
  .cluster-controls label {{ display:flex; align-items:center; gap:6px; }}
  .cluster-controls select {{ font:inherit; }}
  .cluster-charts {{ display:flex; flex-wrap:wrap; gap:28px; align-items:flex-start; }}
  .chart-box {{ background:#fff; border:1px solid #eee; border-radius:6px; padding:14px 16px; }}
  .chart-box h3 {{ font-size:13px; margin:0 0 4px; color:#333; }}
  .chart-box .chart-sub {{ font-size:11px; color:#888; margin:0 0 8px; min-height:14px; }}
  .legend-row {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:10px; font-size:11px; color:#555; }}
  .legend-row .swatch {{ display:inline-block; width:10px; height:10px; border-radius:2px;
                         vertical-align:-1px; margin-right:4px; }}
  .centroid-head {{ display:flex; align-items:center; justify-content:space-between;
                    gap:10px; margin-bottom:4px; }}
  .centroid-head h3 {{ margin:0; }}
  .mean-toggle {{ font:inherit; font-size:11px; padding:3px 9px; border:1px solid #ccc;
                  background:#fff; border-radius:12px; color:#666; cursor:pointer; }}
  .mean-toggle:hover {{ border-color:#999; }}
  .mean-toggle.on {{ background:var(--ink); color:#fff; border-color:var(--ink); }}
  .centroid-row {{ display:flex; flex-wrap:wrap; gap:16px; }}
  .centroid-item {{ text-align:center; }}
  .centroid-item .mark {{ width:130px; height:148px; }}
  .centroid-item .clabel {{ display:flex; align-items:center; justify-content:center;
                            gap:4px; font-size:11px; color:#555; margin-top:3px; }}
  .centroids-box {{ max-width:460px; }}
  .centroid-combined {{ display:flex; flex-direction:column; align-items:stretch; gap:6px;
                        margin-top:16px; padding-top:16px; border-top:1px solid #eee; }}
  .centroid-overlay .mark {{ width:100%; height:auto; aspect-ratio:110/124; }}
  .axis text {{ font:10px -apple-system,Segoe UI,Roboto,sans-serif; fill:#898781; }}
  .axis text.label {{ font-size:11px; fill:#52514e; }}
  .axis .grid {{ stroke:#e1e0d9; stroke-width:1; }}
  .axis .baseline {{ stroke:#c3c2b7; stroke-width:1; }}
  .axis .point {{ stroke:#fff; stroke-width:2; cursor:pointer; }}
  .axis .knee-line {{ fill:none; stroke:var(--knee); stroke-width:2; }}
  .axis .knee-dot {{ fill:var(--knee); stroke:#fff; stroke-width:2; }}
  .axis .knee-dot.selected {{ fill:var(--ink); }}
</style></head>
<body>
  <div class="layout">
    <div class="table-col">
      <h1>Maker&rsquo;s Mark samples vs. mean</h1>
      <p class="sub">PCA basis fitted over {basis.n_seeds} seed marks ({basis.dim} features,
      {basis.n_components} components). Each row is one sample from <code>Samples/</code>:
      its mark overlaid on the population <b>mean</b> (grey), how far it sits from the mean
      in feature space (&Delta;, the L2 distance whose square splits exactly across the
      columns), and the eigen-axes ({("PC" + str(n_comp)) if n_comp < basis.n_components else "all " + str(n_comp) + " components"}
      shown) that carry the most of that deviation. Each heatmap cell is that sample's
      coefficient on one principal component, in standard deviations (&sigma;) of how much
      the seed population itself varies along that axis &mdash; hover a cell for the exact
      coefficient and its share of the sample's total deviation, or click it to see the
      sample with that component's contribution subtracted back out, in the panel to the
      right. Column headers show what share of the population's total variance that axis
      explains. Rows are sorted by
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
    </div>
    <div class="big-col">
      <div class="bigpanel">
        <div id="bigmark"></div>
        <div class="bigcaption" id="bigcaption">Click a sample&rsquo;s thumbnail (left
        column) for a mean-vs-sample comparison, or click one of its heatmap cells to see
        that sample with the component subtracted back out.</div>
      </div>
    </div>
  </div>
  {cluster_section}
  <script>
{script}
  </script>
</body></html>
"""


_SCRIPT_TEMPLATE = r"""
const DATA = __DATA__;
const CLUSTER_COLORS = __CLUSTER_COLORS__;
const MAX_K = __MAX_K__;
const STROKE_WIDTH = __STROKE_WIDTH__;
const VIEWBOX = "__VIEWBOX__";
const N_COMP = __N_COMP__;
const HAS_CLUSTERS = __HAS_CLUSTERS__;

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

function markSvg(layers) {
  const body = layers.map(L => L.ds.map(d =>
    `<path d="${d}" fill="none" stroke="${L.color}" stroke-width="${L.width}" stroke-miterlimit="10" opacity="${L.opacity ?? 1}"/>`
  ).join("")).join("");
  return `<svg viewBox="${VIEWBOX}" preserveAspectRatio="xMidYMid meet" class="mark">${body}</svg>`;
}

function renderSample(i) {
  const row = DATA.rows[i];
  document.getElementById('bigmark').innerHTML = markSvg([
    {ds: DATA.meanDs, color: '#bbbbbb', width: STROKE_WIDTH},
    {ds: row.full, color: '#231f20', width: STROKE_WIDTH},
  ]);
  document.getElementById('bigcaption').innerHTML =
    `<b>${escapeHtml(row.name)}</b> vs. population mean &middot; full sample &middot; &Delta;=${row.dist.toFixed(2)}`;
}

function renderComponent(i, k) {
  const row = DATA.rows[i];
  const comp = row.comps[k];
  document.getElementById('bigmark').innerHTML = markSvg([
    {ds: row.full, color: '#231f20', width: STROKE_WIDTH},
    {ds: comp.d, color: comp.color, width: STROKE_WIDTH},
  ]);
  document.getElementById('bigcaption').innerHTML =
    `<b>${escapeHtml(row.name)}</b> with PC${k + 1} subtracted out ` +
    `(z=${comp.z.toFixed(2)}&sigma;, coeff=${comp.coeff.toFixed(3)}, ${comp.pct.toFixed(1)}% of this ` +
    `sample's total deviation) &middot; ink = full sample, tint = sample minus that component`;
}

document.querySelectorAll('table tbody').forEach(tbody => {
  tbody.addEventListener('click', e => {
    const cell = e.target.closest('td.cell[data-col]');
    if (cell) { renderComponent(+cell.dataset.row, +cell.dataset.col); return; }
    const label = e.target.closest('th.rowlabel[data-row]');
    if (label) { renderSample(+label.dataset.row); }
  });
});

if (DATA.rows.length) renderSample(0);

if (HAS_CLUSTERS) {
  // Decode a full coefficient vector to path "d" strings -- ports
  // eigen.decode + unflatten + genome.PathGene.to_d, same as eigen_explorer.py's
  // client-side decoder. Used to turn a cluster's averaged coefficient vector
  // (DATA.rows[i].coeffsFull) back into a drawable centroid mark.
  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  function decodePaths(coeffs) {
    const dim = DATA.meanFeat.length, s = DATA.meanFeat.slice();
    for (let k = 0; k < coeffs.length; k++) {
      const c = coeffs[k]; if (!c) continue;
      const pc = DATA.components[k];
      for (let j = 0; j < dim; j++) s[j] += c * pc[j];
    }
    const feat = new Array(dim);
    for (let j = 0; j < dim; j++) feat[j] = s[j] / DATA.weights[j];

    let i = 0; const ds = [];
    for (const pid of DATA.pathOrder) {
      const nNodes = DATA.segments[pid] + 1;
      const nodes = [];
      for (let n = 0; n < nNodes; n++) {
        nodes.push([clamp(feat[i], 0, DATA.canvasW), clamp(feat[i + 1], 0, DATA.canvasH)]);
        i += 2;
      }
      const startHandle = [feat[i], feat[i + 1]], endHandle = [feat[i + 2], feat[i + 3]];
      i += 4;
      const tangents = [];
      for (let t = 0; t < nNodes - 2; t++) {
        const theta = feat[i], li = feat[i + 1], lo = feat[i + 2]; i += 3;
        tangents.push([Math.cos(theta), Math.sin(theta),
                       Math.max(DATA.minHandle, li), Math.max(DATA.minHandle, lo)]);
      }
      ds.push(toD(nodes, startHandle, endHandle, tangents));
    }
    return ds;
  }

  function toD(nodes, startHandle, endHandle, tangents) {
    const last = nodes.length - 1;
    const outHandle = (idx) => idx === 0 ? startHandle
          : [tangents[idx - 1][0] * tangents[idx - 1][3], tangents[idx - 1][1] * tangents[idx - 1][3]];
    const inHandle = (idx) => {
      const j = idx + 1;
      if (j === last) return endHandle;
      const tg = tangents[j - 1];
      return [-tg[0] * tg[2], -tg[1] * tg[2]];
    };
    const f = (v) => { let x = v.toFixed(2).replace(/\.?0+$/, ''); return x === '-0' || x === '' ? '0' : x; };
    const p0 = nodes[0];
    let out = 'M' + f(p0[0]) + ',' + f(p0[1]);
    for (let seg = 0; seg < last; seg++) {
      const a = nodes[seg], b = nodes[seg + 1], oh = outHandle(seg), ih = inHandle(seg);
      const c1 = [a[0] + oh[0], a[1] + oh[1]], c2 = [b[0] + ih[0], b[1] + ih[1]];
      out += 'C' + f(c1[0]) + ',' + f(c1[1]) + ',' + f(c2[0]) + ',' + f(c2[1])
           + ',' + f(b[0]) + ',' + f(b[1]);
    }
    return out;
  }

  function seededRng(seed) {
    let s = seed >>> 0;
    return function() {
      s |= 0; s = (s + 0x6D2B79F5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function dist2(a, b) { const dx = a[0] - b[0], dy = a[1] - b[1]; return dx * dx + dy * dy; }

  function kmeansOnce(points, k, rng) {
    const n = points.length;
    k = Math.min(k, n);
    const centers = [points[Math.floor(rng() * n)].slice()];
    while (centers.length < k) {
      const d2 = points.map(p => Math.min(...centers.map(c => dist2(p, c))));
      const sum = d2.reduce((a, b) => a + b, 0);
      if (sum <= 0) { centers.push(points[centers.length % n].slice()); continue; }
      let r = rng() * sum, idx = 0;
      for (; idx < n - 1; idx++) { r -= d2[idx]; if (r <= 0) break; }
      centers.push(points[idx].slice());
    }
    let assign = new Array(n).fill(-1);
    for (let iter = 0; iter < 100; iter++) {
      let changed = false;
      for (let i = 0; i < n; i++) {
        let best = 0, bestD = Infinity;
        for (let c = 0; c < k; c++) { const d = dist2(points[i], centers[c]); if (d < bestD) { bestD = d; best = c; } }
        if (assign[i] !== best) { assign[i] = best; changed = true; }
      }
      const sums = Array.from({length: k}, () => [0, 0, 0]);
      for (let i = 0; i < n; i++) { const c = assign[i]; sums[c][0] += points[i][0]; sums[c][1] += points[i][1]; sums[c][2]++; }
      for (let c = 0; c < k; c++) { if (sums[c][2] > 0) centers[c] = [sums[c][0] / sums[c][2], sums[c][1] / sums[c][2]]; }
      if (!changed) break;
    }
    let inertia = 0;
    for (let i = 0; i < n; i++) inertia += dist2(points[i], centers[assign[i]]);
    return {assign, centers, inertia};
  }

  function kmeans(points, k, seed) {
    let best = null;
    for (let r = 0; r < 12; r++) {
      const rng = seededRng(seed + r * 104729);
      const res = kmeansOnce(points, k, rng);
      if (!best || res.inertia < best.inertia) best = res;
    }
    return best;
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

  function seedFor(xk, yk, k) { return 1000003 * (xk + 1) + 7919 * (yk + 1) + 97 * k; }

  function getPoints(xk, yk) {
    return DATA.rows.map(row => [row.comps[xk].coeff, row.comps[yk].coeff]);
  }

  const SCATTER_W = 460, SCATTER_H = 380, SCATTER_M = {left: 46, right: 16, top: 16, bottom: 40};

  function renderScatter() {
    const xk = +document.getElementById('axisX').value;
    const yk = +document.getElementById('axisY').value;
    const k = +document.getElementById('kSlider').value;
    document.getElementById('kLabel').textContent = k;
    const pts = getPoints(xk, yk);
    const {assign, inertia} = kmeans(pts, k, seedFor(xk, yk, k));

    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    const xPad = (Math.max(...xs) - Math.min(...xs)) * 0.12 || 1;
    const yPad = (Math.max(...ys) - Math.min(...ys)) * 0.12 || 1;
    const x0 = Math.min(...xs) - xPad, x1 = Math.max(...xs) + xPad;
    const y0 = Math.min(...ys) - yPad, y1 = Math.max(...ys) + yPad;
    const sx = scaleLinear(x0, x1, SCATTER_M.left, SCATTER_W - SCATTER_M.right);
    const sy = scaleLinear(y0, y1, SCATTER_H - SCATTER_M.bottom, SCATTER_M.top);

    const xt = niceTicks(x0, x1, 5), yt = niceTicks(y0, y1, 5);
    let svg = '';
    xt.forEach(t => {
      const px = sx(t);
      svg += `<line class="grid" x1="${px}" x2="${px}" y1="${SCATTER_M.top}" y2="${SCATTER_H - SCATTER_M.bottom}"/>`;
      svg += `<text x="${px}" y="${SCATTER_H - SCATTER_M.bottom + 14}" text-anchor="middle">${t}</text>`;
    });
    yt.forEach(t => {
      const py = sy(t);
      svg += `<line class="grid" x1="${SCATTER_M.left}" x2="${SCATTER_W - SCATTER_M.right}" y1="${py}" y2="${py}"/>`;
      svg += `<text x="${SCATTER_M.left - 8}" y="${py + 3}" text-anchor="end">${t}</text>`;
    });
    svg += `<line class="baseline" x1="${SCATTER_M.left}" x2="${SCATTER_W - SCATTER_M.right}" y1="${SCATTER_H - SCATTER_M.bottom}" y2="${SCATTER_H - SCATTER_M.bottom}"/>`;
    svg += `<line class="baseline" x1="${SCATTER_M.left}" x2="${SCATTER_M.left}" y1="${SCATTER_M.top}" y2="${SCATTER_H - SCATTER_M.bottom}"/>`;
    svg += `<text class="label" x="${(SCATTER_M.left + SCATTER_W - SCATTER_M.right) / 2}" y="${SCATTER_H - 6}" text-anchor="middle">PC${xk + 1} coefficient</text>`;
    svg += `<text class="label" x="${-(SCATTER_M.top + SCATTER_H - SCATTER_M.bottom) / 2}" y="12" text-anchor="middle" transform="rotate(-90)">PC${yk + 1} coefficient</text>`;

    pts.forEach((p, i) => {
      const color = CLUSTER_COLORS[assign[i] % CLUSTER_COLORS.length];
      const title = `${DATA.rows[i].name}\nPC${xk + 1}=${p[0].toFixed(3)}, PC${yk + 1}=${p[1].toFixed(3)}\ncluster ${assign[i] + 1}`;
      svg += `<circle class="point" data-row="${i}" cx="${sx(p[0])}" cy="${sy(p[1])}" r="6" fill="${color}"><title>${escapeHtml(title)}</title></circle>`;
    });

    document.getElementById('scatter').innerHTML =
      `<svg width="${SCATTER_W}" height="${SCATTER_H}" viewBox="0 0 ${SCATTER_W} ${SCATTER_H}" class="axis">${svg}</svg>`;
    document.getElementById('scatter').querySelectorAll('circle.point').forEach(el => {
      el.addEventListener('click', () => {
        renderSample(+el.dataset.row);
        document.querySelector('.bigpanel').scrollIntoView({behavior: 'smooth', block: 'start'});
      });
    });
    document.getElementById('scatterSub').textContent =
      `k=${k} → inertia ${inertia.toFixed(2)} (within-cluster sum of squared PC${xk + 1}/PC${yk + 1} distances)`;

    const legend = Array.from({length: k}, (_, c) =>
      `<span><span class="swatch" style="background:${CLUSTER_COLORS[c % CLUSTER_COLORS.length]}"></span>Cluster ${c + 1}</span>`
    ).join('');
    document.getElementById('scatterLegend').innerHTML = legend;

    renderKnee(xk, yk, k);
    renderCentroids(k, assign);
  }

  // Cluster centroid = the mean of members' *full* coefficient vectors (every
  // component, not just the plotted xk/yk pair), decoded back to a mark -- so
  // each centroid reflects the whole shape the cluster shares, not just its
  // position on this one axis pair.
  function clusterCentroidDs(k, assign) {
    const nFull = DATA.components.length;
    const centroids = [];
    for (let c = 0; c < k; c++) {
      const members = DATA.rows.filter((_, i) => assign[i] === c);
      if (!members.length) { centroids.push(null); continue; }
      const avg = new Array(nFull).fill(0);
      members.forEach(row => { for (let j = 0; j < nFull; j++) avg[j] += row.coeffsFull[j]; });
      for (let j = 0; j < nFull; j++) avg[j] /= members.length;
      centroids.push({ds: decodePaths(avg), n: members.length});
    }
    return centroids;
  }

  let lastCentroids = [];
  let showCentroidMean = true;

  function renderCentroids(k, assign) {
    lastCentroids = clusterCentroidDs(k, assign);
    drawCentroids();
  }

  function drawCentroids() {
    const centroids = lastCentroids;
    const meanDs = DATA.meanDs;
    const meanLayer = showCentroidMean ? [{ds: meanDs, color: '#bbbbbb', width: STROKE_WIDTH}] : [];

    const row = document.getElementById('centroidRow');
    row.innerHTML = centroids.map((cen, c) => {
      if (!cen) return '';
      const color = CLUSTER_COLORS[c % CLUSTER_COLORS.length];
      const mark = markSvg([...meanLayer, {ds: cen.ds, color, width: STROKE_WIDTH}]);
      return `<div class="centroid-item">${mark}` +
        `<div class="clabel"><span class="swatch" style="background:${color}"></span>` +
        `Cluster ${c + 1} (n=${cen.n})</div></div>`;
    }).join('');

    const overlayLayers = [...meanLayer];
    centroids.forEach((cen, c) => {
      if (cen) overlayLayers.push({ds: cen.ds, color: CLUSTER_COLORS[c % CLUSTER_COLORS.length], width: STROKE_WIDTH});
    });
    document.getElementById('centroidOverlay').innerHTML = markSvg(overlayLayers);
    document.getElementById('centroidOverlaySub').textContent =
      showCentroidMean ? 'all clusters overlaid on the mean (grey)' : 'all clusters overlaid';
  }

  document.getElementById('btnCentroidMean').addEventListener('click', () => {
    showCentroidMean = !showCentroidMean;
    document.getElementById('btnCentroidMean').classList.toggle('on', showCentroidMean);
    drawCentroids();
  });

  const KNEE_W = 340, KNEE_H = 380, KNEE_M = {left: 42, right: 16, top: 16, bottom: 40};

  function renderKnee(xk, yk, kSel) {
    const pts = getPoints(xk, yk);
    const maxK = Math.min(MAX_K, pts.length);
    const inertias = [];
    for (let k = 1; k <= maxK; k++) inertias.push(kmeans(pts, k, seedFor(xk, yk, k)).inertia);

    const x0 = 1, x1 = maxK;
    const y0 = 0, y1 = Math.max(...inertias) * 1.08 || 1;
    const sx = scaleLinear(x0, x1, KNEE_M.left, KNEE_W - KNEE_M.right);
    const sy = scaleLinear(y0, y1, KNEE_H - KNEE_M.bottom, KNEE_M.top);

    let svg = '';
    const yt = niceTicks(y0, y1, 4);
    yt.forEach(t => {
      const py = sy(t);
      svg += `<line class="grid" x1="${KNEE_M.left}" x2="${KNEE_W - KNEE_M.right}" y1="${py}" y2="${py}"/>`;
      svg += `<text x="${KNEE_M.left - 8}" y="${py + 3}" text-anchor="end">${t}</text>`;
    });
    for (let k = 1; k <= maxK; k++) {
      svg += `<text x="${sx(k)}" y="${KNEE_H - KNEE_M.bottom + 14}" text-anchor="middle">${k}</text>`;
    }
    svg += `<line class="baseline" x1="${KNEE_M.left}" x2="${KNEE_W - KNEE_M.right}" y1="${KNEE_H - KNEE_M.bottom}" y2="${KNEE_H - KNEE_M.bottom}"/>`;
    svg += `<line class="baseline" x1="${KNEE_M.left}" x2="${KNEE_M.left}" y1="${KNEE_M.top}" y2="${KNEE_H - KNEE_M.bottom}"/>`;

    const path = inertias.map((v, idx) => `${idx === 0 ? 'M' : 'L'}${sx(idx + 1)},${sy(v)}`).join(' ');
    svg += `<path class="knee-line" d="${path}"/>`;
    inertias.forEach((v, idx) => {
      const k = idx + 1;
      const sel = k === kSel;
      svg += `<circle class="knee-dot${sel ? ' selected' : ''}" cx="${sx(k)}" cy="${sy(v)}" r="${sel ? 6 : 4}"><title>k=${k}: inertia ${v.toFixed(2)}</title></circle>`;
    });
    svg += `<text class="label" x="${(KNEE_M.left + KNEE_W - KNEE_M.right) / 2}" y="${KNEE_H - 6}" text-anchor="middle">clusters (k)</text>`;
    svg += `<text class="label" x="${-(KNEE_M.top + KNEE_H - KNEE_M.bottom) / 2}" y="12" text-anchor="middle" transform="rotate(-90)">inertia</text>`;

    document.getElementById('knee').innerHTML =
      `<svg width="${KNEE_W}" height="${KNEE_H}" viewBox="0 0 ${KNEE_W} ${KNEE_H}" class="axis">${svg}</svg>`;
  }

  const axisX = document.getElementById('axisX');
  const axisY = document.getElementById('axisY');
  for (let k = 0; k < N_COMP; k++) {
    const pct = DATA.varPct[k].toFixed(1);
    axisX.add(new Option(`PC${k + 1} (${pct}% var)`, k, k === 0, k === 0));
    axisY.add(new Option(`PC${k + 1} (${pct}% var)`, k, k === 1, k === 1));
  }
  axisX.addEventListener('change', renderScatter);
  axisY.addEventListener('change', renderScatter);
  document.getElementById('kSlider').addEventListener('input', renderScatter);
  renderScatter();
}
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
    ap.add_argument("--max-k", type=int, default=8, dest="max_k",
                    help="largest cluster count tried in the knee plot (default 8)")
    args = ap.parse_args()

    pop = load_samples()
    basis = PCABasis.fit(pop)
    n_components = args.components if args.components is not None else basis.n_components
    doc = build_html(basis, pop, n_components, args.sort, args.max_k)
    with open(args.out, "w") as fh:
        fh.write(doc)
    top = min(n_components, basis.n_components)
    print(f"fitted {basis.n_components} components from {basis.n_seeds} seeds; "
          f"compared {len(pop)} samples across top {top} to {args.out}")


if __name__ == "__main__":
    main()
