# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Four related pieces sharing one `Samples/` directory:

1. **Image Ranker** — a local web app that shows two images side-by-side for pairwise
   comparison (arrow keys to vote), logging results to `votes.jsonl` for later ranking
   (Elo / Bradley–Terry).
2. **`genome.py`** — a standalone genetic encoding that breeds new Maker's Mark vector
   designs (`Samples/vector_*.svg`) by treating each SVG's Bézier paths as a genome.
3. **Interactive evolver** (`evolve_server.py` / `eigen.py` / `evolve.*`) — a second
   web app on port 8001: a grid of candidate marks, click to pick parents, breed the
   next generation. Breeding happens in a PCA "eigenshape" space (`eigen.py`), not on
   `genome.py`'s raw genes.
4. **Preference learner** (`preference_server.py` / `preference_model.py` /
   `preference.*`) — a third web app on port 8002: a *forced A/B duel* (like the
   ranker) between marks generated from the eigenshape space, with a Bayesian
   *peaked* model — one per **size bucket** — that learns which value along
   *every* eigen-axis is preferred, scheduling duels toward the highest-variance,
   least-settled axes (axis staircase / seed-blend / confirm). `preference_display.py`
   renders the learned preferences and how they evolved. Where the evolver asks
   "which of these do you like", this asks "what eigen-values do you like".

The ranker, evolver, and preference learner are separate apps: the ranker serves
`.png` files on port 8000, the evolver serves bred SVGs on port 8001, the preference
learner serves eigenspace-generated SVGs on port 8002. The eventual intent is to feed
GA-bred candidates into the same ranking pipeline.

## Commands

```bash
python3 server.py            # run the ranking server (add --debug to log requests)
python3 evolve_server.py     # run the interactive evolver on http://127.0.0.1:8001
                             #   --runs-dir DIR to store runs somewhere else (default runs/)
python genome.py             # GA demo: seeds from Samples/, breeds one child -> offspring_demo.svg
python3 eigen.py             # eigenshape demo: fit basis, check round-trip, breed -> eigen_demo.svg
python3 eigen_display.py     # visualize the eigenshape basis -> eigenshapes.html
                             #   -n COMPONENTS / --sigma N / --steps N to tune the walk
python3 sample_display.py    # compare seed samples to the mean, by eigen-axis -> sample_eigen.html
                             #   -n COMPONENTS / --sort distance|file / --max-k N / -o out.html
python3 eigen_explorer.py    # interactive PC-slider tool -> eigen_explorer.html
                             #   -n COMPONENTS / --z-max N / -o out.html
python3 preference_server.py # forced-A/B preference learner on http://127.0.0.1:8002
                             #   --data-dir DIR for session.json + votes.jsonl (default pref_data/)
python3 preference_display.py # visualize learned preferences -> preference_results.html
                             #   --size 40px,60px / -n COMPONENTS / --steps N / --z-max N / --data-dir DIR
python3 preference_model.py  # model demo: recover a synthetic peaked preference
python3 -m unittest test_eigen test_preference -v   # tests for eigen.py + the preference model
pip install -r requirements.txt   # only needed for analyze.ipynb (jupyter/numpy/pandas/matplotlib)
jupyter notebook analyze.ipynb    # Bradley-Terry ranking + bias analysis; Kernel -> Restart & Run All
```

All the Python (`server.py`, `genome.py`, `eigen.py`, `evolve_server.py`,
`eigen_display.py`, `sample_display.py`, `eigen_explorer.py`, `preference_server.py`,
`preference_model.py`, `preference_display.py`) is pure standard library — no install
needed. `Samples/`, `votes.jsonl`, `runs/`, and `pref_data/` are gitignored; they hold
real vote/image/run data, not code. `preference_results.html` is a generated artifact,
left untracked like `eigenshapes.html`, `sample_eigen.html`, and `eigen_explorer.html`.

## Image Ranker architecture (`server.py` / `app.js` / `index.html`)

- `server.py` is a single-file `ThreadingHTTPServer` with no framework. `IMAGES` (the
  list of `.png` filenames in `Samples/`) is captured once at startup — adding/removing
  images requires a restart. Vote filenames are validated against that captured list
  (path traversal isn't possible).
- Two endpoints: `GET /api/pair` (returns two random distinct filenames + a random size
  bucket from `SIZES`) and `POST /api/vote` (appends one JSON line to `votes.jsonl`,
  fsync'd per write).
- `app.js` preloads the *next* pair while the current one is on screen, so voting feels
  instant. `size` affects display only (`style.css`'s `#stage[data-size=…]` rules) and is
  logged with the vote so downstream analysis can check for size-dependent bias.
- Binds to `127.0.0.1` only.

## Genetic algorithm architecture (`genome.py`)

Every `Samples/vector_*.svg` is the *same* drawing — three cubic Bézier paths (`GS`, `D`,
`DG`) with fixed topology (same segment/node counts in every sample; only point locations
vary). `genome.py` parses that fixed structure into a breedable genome and back.

**Canonical gene encoding** — this is the core design decision, done to make crossover
and mutation behave well regardless of how Illustrator originally wrote the path data
(mixed relative/absolute/smooth commands):
- `"node"` genes: on-curve points, stored in **absolute** canvas coordinates (the
  skeleton).
- `"free"` genes: the two path-endpoint handles, stored as vectors **relative to their
  node**.
- `"tangent"` genes: at every *interior* node (where two segments meet smoothly), instead
  of storing two independent handle vectors that crossover/mutation could pull apart into
  a kink, one shared unit tangent direction + two handle lengths (`in`/`out`) is stored.
  Both handles are always rebuilt from that one direction, so a kink is structurally
  unrepresentable no matter what breeding does to the gene.

`PathGene.genes()` / `PathGene.from_genes()` convert between this typed gene list and the
reconstructable path representation; `Genome.to_svg()` renders straight back to the exact
sample template (same viewBox/stroke) with lossless round-tripping.

**Breeding** (`Genome.breed()` = `crossover()` then `mutate()`):
- `crossover()` walks matched gene pairs from two parents and, per gene, either inherits
  one parent's value whole or blends both via **BLX-α** (samples from
  `[min-α·d, max+α·d]`, not the midpoint — children can land outside the parents' range,
  which is what keeps a population from collapsing toward sameness).
- `mutate()` independently perturbs each gene with probability `rate`: nodes/handle
  lengths get Gaussian jitter (`node_sigma`/`handle_sigma`); a tangent gene's direction is
  *rotated* (`angle_sigma`) rather than perturbed as raw coordinates, which is what
  preserves smoothness through mutation.

Because a bred `Genome` is structurally identical to a seed `Genome`, it can itself be a
parent in the next generation — multi-generation breeding requires no special-casing.
`load_samples()` just globs `Samples/vector_*.svg`, so new seed marks are picked up
automatically.

## Eigenshape space (`eigen.py`)

`genome.py`'s per-gene crossover mixes points independently, which destroys the
correlations *between* points and can produce shapes that don't read as the mark.
`eigen.py` is the fix the evolver actually breeds with:

- `flatten()`/`unflatten()` map a `Genome` to/from a 49-scalar feature vector (nodes,
  free handles, interior tangents as *angle* + two lengths; layout fixed by
  `feature_layout()`, `LAYOUT_VERSION` guards persistence compatibility). Angles are
  unwrapped against a reference so they average linearly, and weighted by mean handle
  length so a radian is comparable to a pixel in the PCA.
- `PCABasis.fit()` computes the mean shape + principal components across the seeds
  using the Gram-matrix trick (n×n instead of 49×49) and a pure-stdlib Jacobi
  eigensolver (`jacobi_eigh`). With n seeds the basis has ≤ n−1 components; full rank
  is kept by default so every seed round-trips exactly (`encode()` → `decode()`).
- Breeding (`breed_coeffs` = `crossover_coeffs` + `mutate_coeffs`) operates on the
  coefficient vector. Each coefficient is a whole-shape deformation direction observed
  in the real samples, so offspring stay coherent by construction; mutation is scaled
  per-component by the population std-dev (`basis.stds`).
- `unflatten()` clamps nodes to the canvas and handle lengths to `MIN_HANDLE` — seed
  nodes reach the canvas edge, so it clamps to the full canvas, *not* the margin-3
  clamp `genome.mutate` uses.

## Eigenshape display (`eigen_display.py`)

Renders what the PCA axes *mean* geometrically. `PCABasis.fit()` gives numbers; this
writes a self-contained `eigenshapes.html` where each row is one principal component:
the mark decoded at the population **mean** and stepped ±`--sigma` std-devs along that
single axis (`decode()` with a coefficient vector that's zero except one entry), so a
row isolates that eigenvector's deformation. Blue→grey→red colors the walk, an overlay
column superimposes it, and each row is labeled with the component's variance share and
σ. Refits from `Samples/` on each run (like creating a new evolver run), so it reflects
the current seeds — not any frozen run basis. Output is a generated artifact, left
untracked like `eigen_demo.svg`.

## Sample-vs-mean display (`sample_display.py`)

`eigen_display.py` shows what an axis *means* in isolation; this is the reverse view —
for each real sample in `Samples/`, how far it actually sits from the mean and which
eigen-axes carry that deviation. `PCABasis.encode()` gives an exact per-axis
decomposition of a sample's squared distance from the mean (the components are
orthonormal, so Parseval holds — the z-scores and %-of-deviation figures in the table
are exact, not approximations), rendered as one heatmap row per sample.

The page is interactive, so a single static render carries several views:
- **Layout**: a two-column `.layout` — the table on the left, a "big panel" pinned in
  `.big-col` (`position: sticky`) to its right. Because the sticky element's containing
  block is that row (bounded by the table's height, not the whole page), it stays in
  view while you scroll the table but scrolls away normally once you pass it — it
  doesn't float over the eigenspace section below.
- **Click a sample's thumbnail/name**: the big panel shows that sample (ink) over the
  population mean (grey).
- **Click one of its heatmap cells**: the big panel keeps the *full sample* and shows it
  with that one component's coefficient zeroed out (`residual_coeffs[k] = 0` before
  `decode()`) tinted by sign, so the delta between the two layers is exactly what that
  axis was contributing to this sample.
- All of this geometry (mean, every sample, every sample × shown-component residual) is
  precomputed in Python and embedded as one JSON blob; clicks only swap SVG markup in
  JS, so the page needs no server.

A second section plots samples in eigenspace: pick any two shown components for X/Y,
and a pure-JS k-means (k-means++ init, 12 seeded restarts, deterministic per axis-pair-
and-k so replotting is stable) clusters that 2D projection at a user-chosen `k`, colored
with the dataviz skill's validated categorical palette (`CLUSTER_COLORS`). A knee plot
(inertia vs. `k`, 1..`--max-k`) recomputes for the same axis pair and marks the selected
`k`. Clicking a scatter point jumps the big panel to that sample. Generated artifact,
left untracked like `eigenshapes.html`.

## Eigenshape explorer (`eigen_explorer.py`)

The hands-on counterpart to `eigen_display.py`: where that shows each axis walked
in fixed ±σ steps (a contact sheet), this is a *mixing board* — one slider per
principal component (labeled with its variance share, ranged in std-devs σ), a big
live preview of the decoded mark, and pixel-size controls. Turn several axes at once
and the mark responds; all-zero is the population mean. Each slider carries a
**thumbnail** of what that axis *does* — the mark stepped ±2σ along it and
superimposed blue→grey→red, the same overlay `eigen_display.py` draws (`axisThumb`
reuses `_lerp_color`'s exact coefficients) — so the effect is legible before you drag.
Double-clicking a slider resets that one axis to the mean.

Like the `*_display.py` scripts it writes a **self-contained** HTML with no server:
`PCABasis.decode` is just `mean + Σ cₖ·componentₖ`, so the fitted basis (mean,
components, per-feature `weights`, per-component `stds`) is embedded as one JSON blob
and the browser reproduces the whole pipeline in JS — `decode` → `unflatten` (same
canvas/`MIN_HANDLE` clamps as `eigen.py`) → `PathGene.to_d` (same Bézier
reconstruction and coordinate formatting as `genome.py`). The JS decode is
byte-identical to Python's across the mean, seed round-trips, and random coefficients
(verified against `eigen.decode`), and the SVG download reproduces `Genome.to_svg`
exactly. Sliders are in σ units; the coefficient handed to `decode` is `z·stds[k]`,
matching how `eigen_display.py` walks an axis.

Controls: size buttons resize the big preview (default 480px, the "large display"),
a fixed strip re-renders the mark at 16–64px to check small-size legibility, a mean
overlay toggles a faint grey mean behind the shape, and randomize/reset seed or clear
all axes. **Getting settings out**: a live JSON blob of the slider positions (`z` in
σ units *and* raw `coeffs`) that round-trips — paste it back and press Apply to
restore a mark (accepts either `z` or `coeffs`) — plus a one-click SVG download of the
current shape. Refits from `Samples/` per run; generated artifact, left untracked like
`eigenshapes.html`.

## Evolver app (`evolve_server.py` / `evolve.html` / `evolve.js` / `evolve.css`)

Same stdlib-server pattern as `server.py` (ThreadingHTTPServer, 127.0.0.1, `--debug`).
A **run** lives in `runs/<name>/` as `state.json` + one `gen_NNN/` SVG dir per
generation. `state.json` stores settings, every generation's candidates as PCA
*coefficients* (with parent lineage), and **the fitted `PCABasis` itself** — resuming
never refits, so a run decodes identically even after `Samples/` changes; a
layout-version mismatch is rejected with 409 rather than silently mis-decoding.

Endpoints: `GET/POST /api/runs` (list / create — creating refits the basis from the
current `Samples/`), `GET /api/runs/<name>` (resume latest), `GET .../gen/<g>`
(read-only history), `POST .../breed` (`{selected, n_offspring, elitism}` — parent
pairs are sampled randomly from the selection so large selections don't explode into
all-pairings; one selected parent falls back to mutation-only children; elitism
re-appends the parents as `"elite"` candidates), `POST .../restart` (truncate to gen 0),
`POST /api/runs/load` (`{path}`, open a run stored outside `--runs-dir`). Payloads
inline each candidate's SVG text (decoded on demand from coefficients — the `gen_NNN/`
files are for export/browsing, never read back). Writes are atomic
(`state.json.tmp` + `os.replace`); breeding is serialized per run with a lock.

## Preference model (`preference_model.py`)

Learns, from forced A/B duels, *what value along each eigen-axis is preferred* —
and that the preference is **peaked** (a sweet spot, worse on either side), so
votes can drift back and forth near the optimum without a clear winner. Pure
stdlib, like `eigen.py`.

- **All axes learned, unevenly scheduled:** by default every component is
  active (`n_active` = M = K) and generated candidates vary every axis, rather
  than freezing a tail at the mean. `n_active` remains as a truncation knob
  (used by tests and the demo) for when learning fewer than K axes is wanted —
  tail axes beyond M then sit at the mean, and `phi`/`observe`/`utility`
  truncate incoming (full K-length, as logged) coefficient vectors to M, so
  raising M later loses nothing. What keeps an all-axes model from diluting
  each vote's information across too many weights is scheduling, not
  truncation — see `"axis"` below.
- **Feature map:** standardize by the basis std (`z_k = c_k / stds_k`) and use a
  *per-axis quadratic* map `phi(c) = [z_1..z_M, z_1²..z_M²]`. Utility `U = w·phi`,
  preference `P(a>b) = sigmoid(w·(phi(a)-phi(b)))` (the constant cancels — no
  intercept). The `z²` terms give each axis an interior optimum
  `z* = -w_lin/(2 w_quad)` when `w_quad < 0` (a peak); `w_quad ≥ 0` = *edge*
  preference, flagged as such. `preferred_coeffs()` puts not-yet-curved axes at
  the ±z_max edge (right for exploration); `best_coeffs()` leaves them at the
  mean (right for display — the edge is a flat-prior artifact, not a preference).
- **Posterior:** Bayesian logistic regression, Gaussian prior `N(0, prior_var·I)`,
  fitted by Newton/IRLS **with Armijo backtracking** — the line search is load-
  bearing: once duels concentrate near the optimum the data becomes near-separable
  and an undamped Newton step diverges (weights blow up to ±hundreds). The Laplace
  covariance `N(w_MAP, H^-1)` (Cholesky of the Hessian) is used for Thompson draws
  and for `zstar_stds()` (posterior spread of each axis's z*).
- **Hybrid `next_duel(rng, mode=None)`** returns `(a, b, meta)` and mixes three
  duel kinds. `"axis"` (staircase): every axis held at the current best guess
  except one — chosen by **weighted random sampling** over all M active axes,
  weight = `zstar_stds[k] * stds[k]` (posterior uncertainty of that axis's z*,
  times how much a z-unit on that axis moves the shape) — dueling two points
  that straddle its current z* (bracket width from `zstar_stds`, wide interior
  probe if no peak is known yet). This is what makes high-variance,
  still-uncertain axes get most of the staircase budget while settled axes and
  low-variance tail axes fade out without ever being excluded outright (a
  uniform fallback covers the all-zero/non-finite-scores case, e.g. a totally
  fresh model). One-axis duels lose nothing (the model has no cross terms) and
  concentrate each vote on exactly 2 weights.
  `"blend"`: whole-shape candidates built by Dirichlet-blending 3 real seeds
  (`seed_zs`) over the active axes + jitter, picked by dueling Thompson —
  plausible by construction, covers combinations axis duels can't. `"confirm"`
  (on request only): current best vs. a near neighbour (mean / seed / jittered
  best). `blend_prob()` (née `explore_prob`) anneals `max(0.25, exp(-n/40))`:
  blends dominate early to find where the interior peaks even are, axis duels
  dominate later to sharpen them. The recency buffer (`d_min`, last ~12 marks)
  applies to blend duels; axis/confirm duels intentionally reuse the base point
  and bypass it.

## Preference learner app (`preference_server.py` / `preference.{html,js,css}`)

Same stdlib-server pattern as `server.py` (ThreadingHTTPServer, 127.0.0.1,
`--debug`, port 8002). A **session** lives in `--data-dir` (default `pref_data/`):
`session.json` pins the fitted `PCABasis` plus `n_active` (now always the full
component count — every axis is learned/varied, per `preference_model.py`'s
current philosophy) and `seed_zs` (the seeds' standardized coefficients, for
blend duels), so logged coefficients decode identically forever; `votes.jsonl`
is the append-only log. On startup a compatible session is resumed (a missing
`seed_zs` is migrated in place; a missing or too-low `n_active` — e.g. an old
session from when `--active-var` truncated it — is raised to the full
component count and re-saved, printing a note; this loses nothing since
`votes.jsonl` always logs full K-length coefficient vectors) and **one model
per size bucket** is rebuilt from that bucket's votes (`observe_many`); a
layout-version mismatch exits rather than mis-decoding.

Endpoints: `GET /api/status` (sizes, `votes_by_size`, `n_active`,
`n_components`), `GET /api/next?size=<bucket>[&mode=confirm]` (issue a duel from
that size's model; only "confirm" may be forced — normal traffic lets the model
schedule), `POST /api/vote` (`{duel_id, winner:"a"|"b"|"tie", size, mode?}` →
log it, `observe` on that size's model, and return the **next** duel in the same
response, honoring `mode` again — the UI's Confirm toggle rides this). Issued
duels are held server-side in `DUELS` keyed by `duel_id` as
`(a, b, size, meta)`; a vote whose `size` doesn't match the issued duel is
rejected *without* consuming the duel. Vote records carry the duel's `mode` (and
`axis` for staircase duels) for analysis. `size` scales the inline SVG (a `--h`
CSS var); switching sizes in the UI fetches a fresh duel for the new bucket's
model. `winner:"tie"` (Down arrow) feeds the model as `y=0.5`. One lock guards
`MODELS`/`DUELS`/the log append.

## Preference display (`preference_display.py`)

Like `eigen_display.py` but for *preferences*: reads `pref_data/session.json` +
`votes.jsonl` (pinned basis; any stored `n_active` is ignored — every model
built here learns/varies all components, via `n_active=None`, matching
`preference_model.py`'s current philosophy) and writes a self-contained,
**interactive** `preference_results.html` (like `sample_display.py`: embedded
JSON + JS, no server) with **one section per size bucket** that has votes
(narrow with `--size`). Per section, top to bottom:

- a header pairing the population **mean** mark with the **current best
  guess** (`best_coeffs()` — not `preferred_coeffs()`, whose edge values on
  unresolved axes are artifacts);
- a **Bézier delta view** — best mark in ink over the mean in grey, arrowless
  (the overlay itself is the delta);
- an **evolution filmstrip** (the model refit on vote-log prefixes — one model
  `observe()`d incrementally, IRLS warm-starts making the checkpoints cheap —
  decoded at each checkpoint);
- **per-axis z\* trajectories** (small multiples of z* vs. vote count, shaded
  ±1 `zstar_std`, gaps where an axis had no interior peak rather than fake
  edge values);
- **per-axis utility overlays** — one axis at a time, every step of its walk
  superimposed in a single SVG, stroke-colored on a sequential green ramp
  (`_util_ramp_color`, pale = low learned utility, deep = high, staying in the
  page's existing green="best" hue family per the dataviz skill's
  single-hue-sequential rule) and drawn low-utility-first so the preferred
  shapes sit on top, with the z\* step stroked thicker; labeled
  `z* = x.xx ± spread` with a peak/edge tag and a settled/unsettled tag
  (`SETTLED_ZSTD_STD`), plus a small ramp legend; and
- an **eigenspace scatter** — the seed marks (grey), the population mean (open
  cross at the origin) and the learned preference (`best_coeffs()`, a green
  star) projected onto two user-chosen components (`PC# (x.x% var)` selects,
  reusing `sample_display.py`'s JS helper patterns — `escapeHtml`,
  `niceTicks`, `scaleLinear`, `.axis`/`.grid`/`.baseline` CSS classes), plus an
  off-by-default checkbox that adds every duel candidate this bucket has seen
  (both sides of every vote), colored by duel mode from the dataviz skill's
  categorical palette (skipping its green slot, reserved here for "the learned
  preference"), filled for the winner and hollow for the loser (both hollow on
  a tie), each with a hover tooltip.

Generated artifact, left untracked like `eigenshapes.html`.
