# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Three related pieces sharing one `Samples/` directory:

1. **Image Ranker** — a local web app that shows two images side-by-side for pairwise
   comparison (arrow keys to vote), logging results to `votes.jsonl` for later ranking
   (Elo / Bradley–Terry).
2. **`genome.py`** — a standalone genetic encoding that breeds new Maker's Mark vector
   designs (`Samples/vector_*.svg`) by treating each SVG's Bézier paths as a genome.
3. **Interactive evolver** (`evolve_server.py` / `eigen.py` / `evolve.*`) — a second
   web app on port 8001: a grid of candidate marks, click to pick parents, breed the
   next generation. Breeding happens in a PCA "eigenshape" space (`eigen.py`), not on
   `genome.py`'s raw genes.

The ranker and the evolver are separate apps: the ranker serves `.png` files on port
8000, the evolver serves bred SVGs on port 8001. The eventual intent is to feed
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
python3 -m unittest test_eigen -v   # tests for eigen.py (round-trip, Jacobi, breeding)
pip install -r requirements.txt   # only needed for analyze.ipynb (jupyter/numpy/pandas/matplotlib)
jupyter notebook analyze.ipynb    # Bradley-Terry ranking + bias analysis; Kernel -> Restart & Run All
```

All the Python (`server.py`, `genome.py`, `eigen.py`, `evolve_server.py`,
`eigen_display.py`) is pure standard library — no install needed. `Samples/`, `votes.jsonl`, and `runs/` are
gitignored; they hold real vote/image/run data, not code.

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
