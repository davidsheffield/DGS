"""Interactive eigenshape explorer -- adjust the principal components by hand.

``eigen_display.py`` shows what each principal component *does* by walking one
axis at a time in fixed +/- steps.  This is the hands-on version: a slider per
component, a big live preview of the decoded mark, and controls to view it at
different pixel sizes.  Where ``eigenshapes.html`` is a contact sheet, this is a
mixing board -- turn several axes at once and watch the mark respond.

The decode is pure linear algebra plus the deterministic Bezier reconstruction
from ``genome.py``, so the whole thing runs client-side: the fitted basis (mean,
components, per-feature weights, per-component std-devs) is embedded as JSON and
JavaScript reproduces ``PCABasis.decode`` + ``unflatten`` + ``PathGene.to_d``.
No server -- like ``eigen_display.py`` / ``sample_display.py`` it writes one
self-contained HTML file.  "Get the settings out" is a JSON blob of the slider
positions (in std-dev units and raw coefficients) that can be pasted back in to
restore a mark, plus a one-click SVG download of the current shape.

    python3 eigen_explorer.py                 # all components, +/-3 sigma range
    python3 eigen_explorer.py -n 8 --z-max 4
    python3 eigen_explorer.py -o out.html

Refits from ``Samples/`` on each run, so it reflects the current seeds.  Output
is a generated artifact, left untracked like ``eigenshapes.html``.
"""

from __future__ import annotations

import argparse
import json

from genome import (CANVAS_H, CANVAS_W, MIN_HANDLE, PATH_ORDER, SEGMENTS,
                    STROKE, STROKE_WIDTH, VIEWBOX, load_samples)
from eigen import PCABasis


def _round(seq, nd=6):
    return [round(float(v), nd) for v in seq]


def build_payload(basis: PCABasis, n_components: int, z_max: float) -> dict:
    """Everything the browser needs to decode a coefficient vector itself."""
    total = sum(basis.eigenvalues) or 1.0
    k = min(n_components, basis.n_components)
    return {
        "viewBox": VIEWBOX,
        "canvasW": CANVAS_W,
        "canvasH": CANVAS_H,
        "minHandle": MIN_HANDLE,
        "stroke": STROKE,
        "strokeWidth": STROKE_WIDTH,
        "pathOrder": list(PATH_ORDER),
        "segments": {pid: SEGMENTS[pid] for pid in PATH_ORDER},
        "weights": _round(basis.weights),
        "mean": _round(basis.mean),
        "components": [_round(basis.components[i]) for i in range(k)],
        "stds": _round(basis.stds[:k]),
        "varPct": [round(basis.eigenvalues[i] / total * 100.0, 3) for i in range(k)],
        "nComponents": k,
        "nComponentsTotal": basis.n_components,
        "nSeeds": basis.n_seeds,
        "dim": basis.dim,
        "seedFiles": list(basis.seed_files),
        "topVarPct": round(sum(basis.eigenvalues[:k]) / total * 100.0, 1),
        "zMax": z_max,
        "layoutVersion": basis.layout_version,
    }


# The page is one static string with a JSON payload spliced in.  Keeping it out
# of an f-string means the CSS/JS braces need no escaping.
_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Maker's Mark eigenshape explorer</title>
<style>
  :root { --ink:#231f20; --line:#e3e3e3; --muted:#777; }
  * { box-sizing:border-box; }
  body { font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif; color:var(--ink);
         margin:0; padding:22px 26px; background:#fafafa; }
  h1 { font-size:20px; margin:0 0 4px; }
  .sub { color:var(--muted); margin:0 0 16px; max-width:78ch; }
  .sub b { color:var(--ink); }
  .layout { display:grid; grid-template-columns:minmax(340px,1fr) minmax(320px,420px);
            gap:26px; align-items:start; }
  @media (max-width:820px){ .layout{ grid-template-columns:1fr; } }

  .display-col { position:sticky; top:16px; }
  .toolbar { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-bottom:12px; }
  .toolbar .label { color:var(--muted); font-size:12px; margin-right:2px; }
  button { font:inherit; font-size:13px; padding:5px 11px; border:1px solid #ccc;
           background:#fff; border-radius:6px; color:var(--ink); cursor:pointer; }
  button:hover { border-color:#999; }
  button.on { background:var(--ink); color:#fff; border-color:var(--ink); }
  .spacer { flex:1 1 auto; }

  .stage { display:flex; justify-content:center; padding:18px; background:#fff;
           border:1px solid var(--line); border-radius:10px; }
  .mark { background:#fff; }
  .mark svg { display:block; width:100%; height:100%; }
  .overlay-path { stroke:#c9c9c9; }

  .previews { display:flex; gap:14px; align-items:flex-end; margin-top:14px;
              flex-wrap:wrap; }
  .previews .p { text-align:center; }
  .previews .p .mark { border:1px solid var(--line); border-radius:4px; }
  .previews .cap { color:var(--muted); font-size:11px; margin-top:3px; }

  details.export { margin-top:16px; border:1px solid var(--line); border-radius:8px;
                   background:#fff; padding:0 12px; }
  details.export > summary { cursor:pointer; padding:10px 0; font-weight:600; }
  .export textarea { width:100%; height:150px; font:12px/1.4 ui-monospace,Menlo,Consolas,monospace;
                     border:1px solid var(--line); border-radius:6px; padding:8px;
                     resize:vertical; color:var(--ink); }
  .export .row { display:flex; gap:8px; margin:8px 0 12px; flex-wrap:wrap; }
  .export .note { color:var(--muted); font-size:12px; margin:2px 0 10px; }

  .controls-col h2 { font-size:13px; text-transform:uppercase; letter-spacing:.04em;
                     color:var(--muted); margin:0 0 4px; }
  .hint { color:var(--muted); font-size:12px; margin:0 0 12px; }
  .hint .sw { display:inline-block; width:10px; height:10px; border-radius:2px;
              vertical-align:-1px; margin:0 2px 0 6px; }
  .sliders { display:flex; flex-direction:column; gap:2px; }
  .slider { display:flex; gap:10px; align-items:center; padding:8px 10px;
            border:1px solid var(--line); border-radius:8px; background:#fff; }
  .slider.active { border-color:#b9b9b9; box-shadow:0 0 0 1px #e6e6e6 inset; }
  .slider .thumb { flex:0 0 auto; width:54px; border:1px solid var(--line);
                   border-radius:4px; background:#fff; }
  .slider .thumb svg { display:block; width:100%; height:100%; }
  .slider .body { flex:1 1 auto; min-width:0; }
  .slider .top { display:flex; justify-content:space-between; align-items:baseline; gap:8px; }
  .slider .name { font-weight:700; }
  .slider .var { color:#c0392b; font-size:12px; font-weight:600; }
  .slider .val { font:12px ui-monospace,Menlo,Consolas,monospace; color:var(--muted);
                 min-width:6ch; text-align:right; }
  .slider input[type=range] { width:100%; margin:6px 0 0; accent-color:var(--ink); }
  .barwrap { position:relative; height:3px; background:#eee; border-radius:2px; margin-top:5px; }
  .barwrap .mid { position:absolute; left:50%; top:-1px; width:1px; height:5px; background:#ccc; }
  .barwrap .fill { position:absolute; top:0; height:3px; border-radius:2px; }
  .fill.pos { background:#d92e2e; } .fill.neg { background:#2966d9; }

  .foot { color:var(--muted); font-size:12px; margin-top:18px; }
  .foot b { color:var(--ink); }
</style></head>
<body>
  <h1>Maker&rsquo;s Mark eigenshape explorer</h1>
  <p class="sub" id="sub"></p>

  <div class="layout">
    <div class="display-col">
      <div class="toolbar">
        <span class="label">size</span>
        <span id="sizeButtons"></span>
        <span class="spacer"></span>
        <button id="btnOverlay">mean overlay</button>
        <button id="btnRandom" title="Draw each axis from N(0,1)">randomize</button>
        <button id="btnReset">reset to mean</button>
      </div>
      <div class="stage"><div class="mark" id="bigMark"></div></div>
      <div class="previews" id="previews"></div>

      <details class="export">
        <summary>Get the settings out</summary>
        <p class="note">Live JSON of the slider positions: <code>z</code> in std-dev
          units (what the sliders show) and raw <code>coeffs</code> (decode space).
          Edit and press <b>Apply</b> to restore a mark.</p>
        <textarea id="settings" spellcheck="false"></textarea>
        <div class="row">
          <button id="btnApply">Apply settings</button>
          <button id="btnCopy">Copy JSON</button>
          <button id="btnSvg">Download SVG</button>
        </div>
      </details>
    </div>

    <div class="controls-col">
      <h2>Principal components</h2>
      <p class="hint">Each thumbnail is that axis walked &plusmn;2&sigma; from the mean,
        superimposed like <code>eigenshapes.html</code>:
        <span class="sw" style="background:#2966d9"></span>&minus;&sigma;
        <span class="sw" style="background:#737373"></span>mean
        <span class="sw" style="background:#d92e2e"></span>+&sigma;. Double-click a
        slider to reset it to the mean.</p>
      <div class="sliders" id="sliders"></div>
    </div>
  </div>

  <p class="foot" id="foot"></p>

<script>
const DATA = __DATA__;

// ---- decode: coefficients -> path "d" strings (ports eigen.decode + unflatten
//      + genome.PathGene.to_d) --------------------------------------------------
function clamp(v, lo, hi){ return v < lo ? lo : (v > hi ? hi : v); }

function decodePaths(coeffs){
  const dim = DATA.mean.length, s = DATA.mean.slice();
  for (let k = 0; k < coeffs.length; k++){
    const c = coeffs[k]; if (!c) continue;
    const pc = DATA.components[k];
    for (let j = 0; j < dim; j++) s[j] += c * pc[j];
  }
  const feat = new Array(dim);
  for (let j = 0; j < dim; j++) feat[j] = s[j] / DATA.weights[j];

  // unflatten -> per-path {nodes, startHandle, endHandle, tangents}
  let i = 0; const ds = [];
  for (const pid of DATA.pathOrder){
    const nNodes = DATA.segments[pid] + 1;
    const nodes = [];
    for (let n = 0; n < nNodes; n++){
      nodes.push([clamp(feat[i], 0, DATA.canvasW), clamp(feat[i+1], 0, DATA.canvasH)]);
      i += 2;
    }
    const startHandle = [feat[i], feat[i+1]], endHandle = [feat[i+2], feat[i+3]];
    i += 4;
    const tangents = [];
    for (let t = 0; t < nNodes - 2; t++){
      const theta = feat[i], li = feat[i+1], lo = feat[i+2]; i += 3;
      tangents.push([Math.cos(theta), Math.sin(theta),
                     Math.max(DATA.minHandle, li), Math.max(DATA.minHandle, lo)]);
    }
    ds.push(toD(nodes, startHandle, endHandle, tangents));
  }
  return ds;
}

function toD(nodes, startHandle, endHandle, tangents){
  const last = nodes.length - 1;
  const outHandle = (idx) => idx === 0 ? startHandle
        : [tangents[idx-1][0] * tangents[idx-1][3], tangents[idx-1][1] * tangents[idx-1][3]];
  const inHandle = (idx) => {
    const j = idx + 1;
    if (j === last) return endHandle;
    const tg = tangents[j-1];
    return [-tg[0] * tg[2], -tg[1] * tg[2]];
  };
  const f = (v) => { let x = v.toFixed(2).replace(/\.?0+$/, ''); return x === '-0' || x === '' ? '0' : x; };
  const p0 = nodes[0];
  let out = 'M' + f(p0[0]) + ',' + f(p0[1]);
  for (let s = 0; s < last; s++){
    const a = nodes[s], b = nodes[s+1], oh = outHandle(s), ih = inHandle(s);
    const c1 = [a[0]+oh[0], a[1]+oh[1]], c2 = [b[0]+ih[0], b[1]+ih[1]];
    out += 'C' + f(c1[0]) + ',' + f(c1[1]) + ',' + f(c2[0]) + ',' + f(c2[1])
         + ',' + f(b[0]) + ',' + f(b[1]);
  }
  return out;
}

// blue (neg) -> grey (mean) -> red (pos), matching eigen_display._lerp_color
function lerpColor(t){
  let r, g, b;
  if (t <= 0){ const a = Math.min(1, -t);
    r = (1-a)*0.45 + a*0.16; g = (1-a)*0.45 + a*0.40; b = (1-a)*0.45 + a*0.85;
  } else { const a = Math.min(1, t);
    r = (1-a)*0.45 + a*0.85; g = (1-a)*0.45 + a*0.16; b = (1-a)*0.45 + a*0.18; }
  const h = (x) => Math.round(x*255).toString(16).padStart(2, '0');
  return '#' + h(r) + h(g) + h(b);
}

// Static per-axis illustration: the mark stepped +/-SIG*std along axis k only,
// every step superimposed (the "overlay" column of eigenshapes.html).
function axisThumb(k){
  const SIG = 2, STEPS = 5, mid = (STEPS - 1) / 2;
  let layers = '';
  for (let step = 0; step < STEPS; step++){
    let m = -SIG + 2*SIG*step/(STEPS-1);
    if (step === mid) m = 0;
    const isMean = Math.abs(m) < 1e-9;
    const color = isMean ? DATA.stroke : lerpColor(m/SIG);
    const c = new Array(DATA.nComponents).fill(0);
    c[k] = m * DATA.stds[k];
    const w = isMean ? DATA.strokeWidth : DATA.strokeWidth*0.7;
    layers += decodePaths(c).map(d => `<path d="${d}" fill="none" stroke="${color}" `
      + `stroke-width="${w}" stroke-miterlimit="10" opacity="${isMean ? 1 : 0.75}"/>`).join('');
  }
  return `<svg viewBox="${DATA.viewBox}" preserveAspectRatio="xMidYMid meet">${layers}</svg>`;
}

function markSVG(ds, withOverlay){
  const body = ds.map(d => `<path d="${d}" fill="none" stroke="${DATA.stroke}" `
      + `stroke-width="${DATA.strokeWidth}" stroke-miterlimit="10"/>`).join('');
  let over = '';
  if (withOverlay) over = MEAN_DS.map(d => `<path class="overlay-path" d="${d}" `
      + `fill="none" stroke-width="${DATA.strokeWidth}" stroke-miterlimit="10"/>`).join('');
  return `<svg viewBox="${DATA.viewBox}" preserveAspectRatio="xMidYMid meet">${over}${body}</svg>`;
}

// ---- state ------------------------------------------------------------------
const K = DATA.nComponents;
const z = new Array(K).fill(0);           // slider positions, std-dev units
const MEAN_DS = decodePaths(new Array(K).fill(0));
const AR = (() => { const p = DATA.viewBox.split(/\s+/).map(Number); return p[3] / p[2]; })();
let bigSize = 480, overlay = false;
const SIZES = [128, 256, 384, 480, 640];
const PREVIEWS = [16, 24, 32, 48, 64];

const $ = (id) => document.getElementById(id);

function coeffs(){ return z.map((v, k) => v * DATA.stds[k]); }

function render(){
  const ds = decodePaths(coeffs());
  const big = $('bigMark');
  big.style.width = bigSize + 'px';
  big.style.height = Math.round(bigSize * AR) + 'px';
  big.innerHTML = markSVG(ds, overlay);
  document.querySelectorAll('#previews .mark').forEach(el => { el.innerHTML = markSVG(ds, false); });
  // slider readouts
  for (let k = 0; k < K; k++){
    $('val' + k).textContent = (z[k] >= 0 ? '+' : '') + z[k].toFixed(2) + 'σ';
    const fill = $('fill' + k), frac = clamp(z[k] / DATA.zMax, -1, 1);
    fill.className = 'fill ' + (frac >= 0 ? 'pos' : 'neg');
    fill.style.left = (frac >= 0 ? 50 : 50 + frac * 50) + '%';
    fill.style.width = Math.abs(frac) * 50 + '%';
    $('slider' + k).classList.toggle('active', Math.abs(z[k]) > 0.005);
  }
  $('settings').value = settingsJSON();
}

function settingsJSON(){
  return JSON.stringify({
    n_components: K,
    z: z.map(v => +v.toFixed(4)),
    coeffs: coeffs().map(v => +v.toFixed(5)),
    seed_files: DATA.seedFiles,
    layout_version: DATA.layoutVersion,
  }, null, 2);
}

// ---- build controls ---------------------------------------------------------
function buildSizeButtons(){
  const host = $('sizeButtons');
  SIZES.forEach(sz => {
    const b = document.createElement('button');
    b.textContent = sz + 'px';
    b.onclick = () => { bigSize = sz; markSizeButtons(); render(); };
    b.dataset.sz = sz;
    host.appendChild(b);
  });
  markSizeButtons();
}
function markSizeButtons(){
  document.querySelectorAll('#sizeButtons button').forEach(b =>
    b.classList.toggle('on', +b.dataset.sz === bigSize));
}

function buildPreviews(){
  const host = $('previews');
  PREVIEWS.forEach(px => {
    const wrap = document.createElement('div'); wrap.className = 'p';
    const m = document.createElement('div'); m.className = 'mark';
    m.style.width = px + 'px'; m.style.height = Math.round(px * AR) + 'px';
    const cap = document.createElement('div'); cap.className = 'cap'; cap.textContent = px + 'px';
    wrap.appendChild(m); wrap.appendChild(cap); host.appendChild(wrap);
  });
}

function buildSliders(){
  const host = $('sliders');
  for (let k = 0; k < K; k++){
    const row = document.createElement('div');
    row.className = 'slider'; row.id = 'slider' + k;
    row.innerHTML =
      `<div class="thumb" style="height:${Math.round(54*AR)}px">${axisThumb(k)}</div>`
      + `<div class="body">`
      + `<div class="top"><span><span class="name">PC${k+1}</span> `
      + `<span class="var">${DATA.varPct[k].toFixed(1)}%</span></span>`
      + `<span class="val" id="val${k}">+0.00σ</span></div>`
      + `<input type="range" id="rng${k}" min="${-DATA.zMax}" max="${DATA.zMax}" `
      + `step="0.01" value="0" title="double-click to reset to mean">`
      + `<div class="barwrap"><div class="mid"></div><div class="fill pos" id="fill${k}"></div></div>`
      + `</div>`;
    host.appendChild(row);
    const rng = $('rng' + k);
    rng.addEventListener('input', (e) => { z[k] = +e.target.value; render(); });
    rng.addEventListener('dblclick', () => { z[k] = 0; rng.value = 0; render(); });
  }
}

function setZ(newZ){
  for (let k = 0; k < K; k++){
    z[k] = clamp(newZ[k] || 0, -DATA.zMax, DATA.zMax);
    $('rng' + k).value = z[k];
  }
  render();
}

// ---- toolbar / export actions ----------------------------------------------
function gaussian(){ // Box-Muller
  let u = 0, v = 0; while (u === 0) u = Math.random(); while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function applySettings(){
  try {
    const obj = JSON.parse($('settings').value);
    if (Array.isArray(obj.z)) { setZ(obj.z); return; }
    if (Array.isArray(obj.coeffs)) { setZ(obj.coeffs.map((c, k) => c / DATA.stds[k])); return; }
    alert('JSON needs a "z" or "coeffs" array.');
  } catch (e) { alert('Could not parse settings JSON: ' + e.message); }
}

function downloadSVG(){
  const ds = decodePaths(coeffs());
  const body = DATA.pathOrder.map((pid, i) =>
    `  <path id="${pid}" class="st0" d="${ds[i]}"/>`).join('\n');
  const svg = '<?xml version="1.0" encoding="UTF-8"?>\n'
    + `<svg id="Layer_1" xmlns="http://www.w3.org/2000/svg" version="1.1" viewBox="${DATA.viewBox}">\n`
    + '  <defs>\n    <style>\n      .st0 {\n        fill: none;\n'
    + `        stroke: ${DATA.stroke};\n        stroke-miterlimit: 10;\n`
    + `        stroke-width: ${DATA.strokeWidth}px;\n      }\n    </style>\n  </defs>\n`
    + body + '\n</svg>\n';
  const blob = new Blob([svg], {type: 'image/svg+xml'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'eigen_mark.svg';
  document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(a.href);
}

// ---- wire up ----------------------------------------------------------------
$('sub').innerHTML = `PCA basis fitted over <b>${DATA.nSeeds}</b> seed marks `
  + `(${DATA.dim} features, ${DATA.nComponentsTotal} components). Drag a slider to move `
  + `the mark along that principal component, measured in standard deviations &sigma; from `
  + `the population mean; several at once compose. The ${K} components shown explain `
  + `<b>${DATA.topVarPct}%</b> of the variance across the seeds.`;
$('foot').innerHTML = `Range &plusmn;${DATA.zMax}&sigma; per axis &middot; all-zero = population mean `
  + `&middot; seeds: ${DATA.seedFiles.join(', ')}`;

$('btnReset').onclick = () => setZ(new Array(K).fill(0));
$('btnRandom').onclick = () => setZ(Array.from({length: K}, () => gaussian()));
$('btnOverlay').onclick = () => { overlay = !overlay; $('btnOverlay').classList.toggle('on', overlay); render(); };
$('btnApply').onclick = applySettings;
$('btnCopy').onclick = () => navigator.clipboard.writeText(settingsJSON());
$('btnSvg').onclick = downloadSVG;

buildSizeButtons();
buildPreviews();
buildSliders();
render();
</script>
</body></html>
"""


def build_html(basis: PCABasis, n_components: int, z_max: float) -> str:
    payload = build_payload(basis, n_components, z_max)
    return _PAGE.replace("__DATA__", json.dumps(payload))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="eigen_explorer.html",
                    help="output HTML file (default eigen_explorer.html)")
    ap.add_argument("-n", "--components", type=int, default=None,
                    help="number of top principal components to expose (default: all)")
    ap.add_argument("--z-max", type=float, default=3.0,
                    help="slider range each side, in std-devs (default 3)")
    args = ap.parse_args()

    pop = load_samples()
    basis = PCABasis.fit(pop)
    n_components = args.components if args.components is not None else basis.n_components
    doc = build_html(basis, n_components, args.z_max)
    with open(args.out, "w") as fh:
        fh.write(doc)
    shown = min(n_components, basis.n_components)
    print(f"fitted {basis.n_components} components from {basis.n_seeds} seeds; "
          f"wrote explorer with {shown} sliders to {args.out}")


if __name__ == "__main__":
    main()
