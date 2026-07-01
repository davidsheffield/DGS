"""Visualize what ``preference_server.py`` learned about the eigenshape space.

``eigen_display.py`` shows what each PCA axis *does*; this shows which values
along those axes you *prefer*.  It reads a session (``pref_data/session.json``
for the pinned basis) and its logged duels (``pref_data/votes.jsonl``), refits
the Bayesian peaked model (``preference_model.py``), and writes a self-contained
``preference_results.html``:

* a header pairing the population **mean** mark with the **preferred** mark
  (the model's overall optimum decoded back to a drawing), and
* one row per eigen-axis -- the mark stepped across that axis, each cell tinted
  by the learned utility, the **preferred value z\\*** highlighted, and a little
  utility profile bar under the walk -- sorted so the axes your votes most
  strongly constrain come first.

Refits from the log on each run, like ``eigen_display.py`` refits from
``Samples/``.  Pure standard library; output is a generated artifact.

    python3 preference_display.py                 # all axes, +/-2.5 std, 9 steps
    python3 preference_display.py -n 8 --steps 11
    python3 preference_display.py --data-dir DIR -o out.html
"""

from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

from eigen import PCABasis
from genome import PATH_ORDER, STROKE_WIDTH, VIEWBOX
from preference_model import WINNER_Y, PreferenceModel

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "pref_data"


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
    return state, basis, votes


# ---------------------------------------------------------------------------
# Rendering helpers
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


def build_html(state: dict, basis: PCABasis, model: PreferenceModel,
               votes: list[dict], n_components: int, z_max: float,
               steps: int) -> str:
    if steps < 3:
        steps = 3
    if steps % 2 == 0:
        steps += 1
    half = steps // 2
    zs = [(-z_max + 2 * z_max * k / (steps - 1)) for k in range(steps)]
    zs[half] = 0.0

    n_ties = sum(1 for v in votes if v["winner"] == "tie")
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

        # profile column: preferred mark for this axis at z_star, plus a summary
        star_ds = _paths_at_coeffs(
            basis, [z_star * std if j == k else 0.0 for j in range(basis.n_components)])
        kind = "peak" if r["peak"] else "edge"
        cells.append(
            f'<td class="cell profile">{_svg(star_ds, STROKE_WIDTH)}'
            f'<div class="ptag">{kind} @ {z_star:+.2f}σ</div></td>')

        label = (f'<div class="pc">PC{k + 1}</div>'
                 f'<div class="star">{z_star:+.2f}σ</div>'
                 f'<div class="kind {kind}">{kind}</div>')
        rows.append(f'<tr><th class="rowlabel">{label}</th>{"".join(cells)}</tr>')

    mean_ds = _paths_at_coeffs(basis, [0.0] * basis.n_components)
    pref_ds = _paths_at_coeffs(basis, model.preferred_coeffs())
    seeds = html.escape(", ".join(basis.seed_files))
    note = ("" if model.n_obs >= 15 else
            '<p class="warn">Only %d votes so far — the model is still close to '
            'its prior; keep voting for the preferred values to sharpen.</p>'
            % model.n_obs)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Maker's Mark preferences</title>
<style>
  :root {{ --ink:#231f20; }}
  body {{ font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif; color:var(--ink);
         margin:0; padding:24px 28px; background:#fafafa; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  .sub {{ color:#666; margin:0 0 16px; max-width:74ch; }}
  .warn {{ color:#b8860b; margin:0 0 16px; }}
  .summary {{ display:flex; gap:28px; align-items:center; margin:0 0 22px; }}
  .summary figure {{ margin:0; text-align:center; }}
  .summary figcaption {{ color:#666; font-size:12px; margin-top:4px; }}
  .big {{ width:150px; height:170px; background:#fff; border:1px solid #ddd;
          border-radius:6px; }}
  .arrow {{ font-size:26px; color:#1a7f37; }}
  table {{ border-collapse:collapse; }}
  thead th {{ font-weight:600; color:#555; font-size:12px; padding:4px 0; text-align:center; }}
  thead th:first-child {{ width:74px; }}
  .rowlabel {{ text-align:left; padding:0 10px 0 2px; vertical-align:middle; white-space:nowrap; }}
  .rowlabel .pc {{ font-weight:700; font-size:15px; }}
  .rowlabel .star {{ color:#1a7f37; font-size:13px; font-weight:600; }}
  .rowlabel .kind {{ font-size:11px; color:#888; }}
  .rowlabel .kind.edge {{ color:#c0392b; }}
  .cell {{ vertical-align:middle; text-align:center; border:1px solid #eee; }}
  .cell.star {{ outline:2px solid #1a7f37; outline-offset:-2px; }}
  .mark {{ display:block; width:82px; height:92px; margin:2px auto 0; }}
  .bar {{ height:26px; display:flex; align-items:flex-end; justify-content:center; }}
  .bar i {{ display:block; width:60%; background:#1a7f37; opacity:.55; }}
  .cell.profile {{ background:#f4fbf6; border-left:2px solid #cfe9d8; }}
  .cell.profile .ptag {{ font-size:11px; color:#1a7f37; padding:2px 0 4px; }}
</style></head>
<body>
  <h1>Maker&rsquo;s Mark preferences</h1>
  <p class="sub">Bayesian peaked model fitted to <b>{model.n_obs}</b> logged duels
  ({n_ties} ties) over the eigenshape basis ({basis.n_seeds} seeds,
  {basis.n_components} components). Each row is one eigen-axis: the mark stepped
  across it (all other axes at the mean), each cell tinted and barred by the
  learned utility, and the preferred value <b>z*</b> outlined in green. Rows are
  sorted by how strongly your votes constrain the axis.</p>
  {note}
  <div class="summary">
    <figure>{_svg(mean_ds, STROKE_WIDTH, "big")}<figcaption>population mean</figcaption></figure>
    <div class="arrow">&rarr;</div>
    <figure>{_svg(pref_ds, STROKE_WIDTH, "big")}<figcaption>preferred (model optimum)</figcaption></figure>
  </div>
  <table>
    <thead><tr><th></th>{col_head}</tr></thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
  <p class="sub" style="margin-top:16px">
    <b>peak</b> = an interior sweet spot (utility falls off on both sides);
    <b class="warn" style="color:#c0392b">edge</b> = preference keeps rising to the
    &plusmn;{z_max:g}σ bound, no interior optimum. &nbsp;·&nbsp; seeds: {seeds}
  </p>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="preference_results.html",
                    help="output HTML file (default preference_results.html)")
    ap.add_argument("-n", "--components", type=int, default=999,
                    help="max eigen-axes to show (default: all)")
    ap.add_argument("--data-dir", default=str(DATA_DIR),
                    help="session directory (default pref_data/)")
    ap.add_argument("--z-max", type=float, default=2.5,
                    help="std-devs to walk each side of the mean")
    ap.add_argument("--steps", type=int, default=9,
                    help="cells per row across the walk (odd; mean centered)")
    args = ap.parse_args()

    state, basis, votes = load_session(Path(args.data_dir).expanduser().resolve())
    model = PreferenceModel(basis.stds, z_max=args.z_max, rng=random.Random(0))
    model.observe_many(votes)
    doc = build_html(state, basis, model, votes, args.components, args.z_max, args.steps)
    Path(args.out).write_text(doc, encoding="utf-8")
    print(f"fitted preferences from {len(votes)} duels; wrote {args.out}")


if __name__ == "__main__":
    main()
