# Image Ranker

A tiny local web app for ranking images by pairwise comparison. Two images from `Samples/` are shown side-by-side on a white background; you pick the one you prefer with the arrow keys. Each vote is appended to `votes.jsonl` so rankings can be computed offline later (e.g. Elo or Bradley–Terry).

## Requirements

- Python 3.9+ (standard library only — no packages to install)
- A modern browser

## Layout

```
comparer/
├── Samples/        # put the PNG images you want to rank here
├── server.py       # local HTTP server + vote logger
├── index.html      # page shell
├── style.css       # white background, centered flex layout
├── app.js          # pair fetching, keyboard handling, vote POST
└── votes.jsonl     # created on first vote, one JSON object per line
```

## Run

From the project directory:

```bash
python3 server.py
```

You should see:

```
serving on http://127.0.0.1:8000 — 38 images loaded
```

Open http://127.0.0.1:8000 in a browser.

Stop the server with `Ctrl-C`. Votes are flushed to disk as they happen, so you can stop and resume at any time.

## Use

- Two images appear side-by-side.
- Press **`←`** if you prefer the **left** image.
- Press **`→`** if you prefer the **right** image.
- The next pair appears immediately.

That is the entire interaction. No other controls, no on-screen text — the page is deliberately minimal so nothing biases the comparison.

Behind the scenes, each pair is rendered at one of three sizes (small / medium / large), chosen randomly per pair. The size isn't shown but is recorded with the vote so you can analyze size-dependent effects later.

## Vote log format

`votes.jsonl` — one JSON object per line, append-only:

```json
{"ts":"2026-04-19T14:32:30Z","pair_id":"17eabe79…","a":"11.png","b":"19.png","size":"large","winner":"a"}
```

Fields:
- `ts` — UTC timestamp when the vote was recorded
- `pair_id` — unique id for the pair that was shown (UUID4 hex)
- `a`, `b` — filenames of the left and right images
- `size` — `small` (20 px tall), `medium` (40 px), or `large` (60 px)
- `winner` — `"a"` or `"b"`

## Computing rankings

The log format is trivial to load. Example with Python:

```python
import json
votes = [json.loads(l) for l in open("votes.jsonl")]
```

From there, feed the pairs into your favorite ranking method — Elo, Bradley–Terry, TrueSkill, or a simple win-rate sort.

### Analysis notebook

`analyze.ipynb` contains a ready-made analysis of `votes.jsonl`: pair coverage, size distribution, left/right position bias with ±1σ error bars, and Bradley–Terry rankings overall and per size.

Requirements (in addition to Python 3.9+):

```bash
pip install -r requirements.txt
```

Run it:

```bash
jupyter notebook analyze.ipynb
```

Then *Kernel → Restart & Run All*. The notebook re-reads `votes.jsonl` from the project directory on every run, so re-executing picks up any new votes.

## Evolving new marks (`genome.py`)

`genome.py` is a self-contained genetic algorithm for the vector marks in
`Samples/vector_*.svg`. Each of those SVGs is the same drawing — three cubic
Bézier paths (`GS`, `D`, `DG`) with **fixed topology** (the same number of
nodes everywhere); only the point locations differ. The module treats each SVG
as a genome so new marks can be bred from old ones and rendered back to SVG.

Pure standard library (`math` + `random`) — no packages to install — so the
server can reuse it later.

### Gene encoding

Illustrator writes the paths with a mix of relative (`c`), absolute (`C`) and
smooth (`s`) commands just to save bytes, which is what makes "are the control
points relative or absolute?" a confusing question. `genome.py` removes the
ambiguity by parsing every command into one canonical form and storing:

- **nodes** — on-curve points in **absolute** canvas coordinates (the skeleton);
- **endpoint handles** — the free handle at the start of each path (and both
  handles of the single-segment `D`) as a **vector relative to its node**;
- **interior-node tangents** — at every node where two segments meet, the two
  handles in the source are exactly antiparallel (the curve is *smooth*). Rather
  than store two vectors that could drift apart, a node stores **one shared
  tangent direction + two handle lengths**.

Storing interior nodes as a single direction makes a kink *unrepresentable*:
both handles are always rebuilt from the same unit vector, so no amount of
crossover or mutation can introduce the cusps you'd get from perturbing two
independent handles. A handle stays meaningful regardless of which parent's node
it ends up attached to, which is what makes recombination behave. Internally a
genome is a flat, typed gene list (`"node"` / `"free"` / `"tangent"`); breeding
never touches SVG syntax.

Fixed topology, identical across every sample:

| path | cubic segments | on-curve nodes |
|------|----------------|----------------|
| `GS` | 4              | 5              |
| `DG` | 3              | 4              |
| `D`  | 1              | 2              |

### Breeding two parents

```python
import genome as G

pop = G.load_samples()              # one Genome per Samples/vector_*.svg
child = pop[0].breed(pop[3], rate=0.15)   # crossover, then mutate
child.save_svg("offspring.svg")     # back to the exact sample template
```

`breed()` is the single call the selection driver makes once two parents are
chosen; it is just `crossover()` followed by `mutate()`. Because each parent is
itself a `Genome` produced the same way, a parent already carries the mixed
genes of *its* parents — the multi-generation flow comes for free.

**Crossover.** For each component, per path, independently either *inherit* the
whole point from one random parent (keeps a coherent feature intact) or *blend*
the two parents with **BLX-α**. BLX-α draws from `[min − α·d, max + α·d]` rather
than the midpoint, so children can land *outside* the parents' range — that is
the antidote to "averaging makes every offspring look the same." Knobs:
`blend_prob` (default `0.5`), `alpha` (default `0.5`).

**Mutation.** `mutate(rate, …)` gives each gene an independent chance of a
Gaussian nudge. `rate` is the mutation probability (e.g. `0.15` ≈ 15 % of genes
move). Magnitudes are per gene type: `node_sigma` (default `4` px) moves the
skeleton, `handle_sigma` (default `6` px) jitters handle lengths, and
`angle_sigma` (default `0.15` rad) *rotates* an interior node's shared tangent —
which reshapes the curl while keeping the node smooth. Nodes are clamped to the
canvas; handles may point outside it but never shrink below a small floor.

The defaults (`blend_prob`, `alpha`, `node_sigma`, `handle_sigma`, `angle_sigma`)
are reasonable starting points and worth tuning once a generation is rendered.

### Rendering genes to SVG

`Genome.to_svg()` emits the exact sample template (`viewBox 0 0 110 124`,
stroke `#231f20`, 3 px) using absolute `C` commands — ready to display as-is, no
rasterization needed. Parsing then re-rendering a sample is lossless, and bred
offspring always re-parse to valid genomes with the same topology.

### Demo

```bash
python genome.py        # seeds from Samples/, breeds one child -> offspring_demo.svg
```

The seed list is just a glob of `Samples/vector_*.svg`, so it works as more of
the first generation is added.

## Interactive evolution (`evolve_server.py` + `eigen.py`)

A second local web app that drives the GA interactively: it shows a grid of
candidate marks, you click the ones you like, and the server breeds the next
generation from your picks.

```bash
python3 evolve_server.py     # then open http://127.0.0.1:8001
```

Also pure standard library. Options: `--port`, `--debug`, `--runs-dir DIR`
(where runs are stored, default `runs/`), `--samples DIR`, `--var-keep`.

### Why not breed with `genome.py` directly?

`genome.py`'s crossover treats every node/handle as an independent gene.
Mixing points one at a time destroys the correlations *between* points that
make the mark read as one drawing, so offspring can come out bizarre.

`eigen.py` fixes this with a PCA **eigenshape** encoding: every genome flattens
to a 49-number feature vector, and the mean shape + principal components are
computed across the seed population (pure stdlib — Gram-matrix trick plus a
Jacobi eigensolver). Breeding then happens on the ~21 PCA *coefficients*
instead of raw points. Each coefficient is a whole-shape deformation direction
observed in the real samples, so crossover and mutation can only move points
together, the way they move in actual marks — incoherent shapes are nearly
unrepresentable. Crossover is per-coefficient inherit-or-BLX-α; mutation is
Gaussian, scaled by each component's population spread.

### Using the app

- **Generation 0** is the seed population from `Samples/vector_*.svg`.
- Click cards to select the parents you want to propagate; the **offspring**
  number controls how many children are bred (parent pairs are sampled at
  random from your selection, so a big selection doesn't explode into
  all-pairings). One selected parent works too — its children are mutants.
- **keep parents** (elitism, on by default) carries your selected parents into
  the next generation so favorites can't be lost to one bad batch.
- **← / →** browse earlier generations read-only (past picks shown in orange);
  breeding always continues from the latest generation.
- **New run…** starts a fresh run (recomputes the PCA basis from whatever is
  in `Samples/` at that moment) and can write it to a custom directory.
  **Restart** deletes every bred generation of the current run and returns it
  to generation 0.

### Run storage

Each run lives in `runs/<name>/` (or wherever you pointed it): `state.json`
holds the settings, every generation's coefficient vectors with parent lineage,
and the fitted PCA basis itself — so a run resumes and decodes *identically*
even if `Samples/` gains new files later. Each generation's SVGs are also
written to `gen_NNN/` for browsing/export. Stop the server any time; reopening
a run picks up exactly where it left off. A run stored elsewhere can be opened
via `POST /api/runs/load {"path": …}`.

Tests: `python3 -m unittest test_eigen -v`.

## Adding or removing images

Drop `.png` files into `Samples/` (or remove them) and restart the server. The image list is captured at startup; the server ignores anything that isn't a `.png` file in that directory.

## Configuration

The constants near the top of `server.py` are the only knobs:

- `HOST`, `PORT` — where the server binds (default `127.0.0.1:8000`)
- `SIZES` — the three size buckets used for display
- The pixel heights for each size live in `style.css` under `#stage[data-size=…]`

## Notes

- The server binds to `127.0.0.1` only — not reachable from the network.
- Pairs are drawn uniformly at random with replacement, so the same pair can recur. That's intentional — repeated comparisons improve ranking stability.
- Filenames in vote requests are validated against the startup directory listing, so path traversal isn't possible.
