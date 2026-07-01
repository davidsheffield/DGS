# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Two related pieces sharing one `Samples/` directory:

1. **Image Ranker** — a local web app that shows two images side-by-side for pairwise
   comparison (arrow keys to vote), logging results to `votes.jsonl` for later ranking
   (Elo / Bradley–Terry).
2. **`genome.py`** — a standalone genetic algorithm that breeds new Maker's Mark vector
   designs (`Samples/vector_*.svg`) by treating each SVG's Bézier paths as a genome.

The two aren't wired together yet: `genome.py` doesn't call the server, and the server
only serves `.png` files. The eventual intent is for the server to serve GA-bred
candidates into the same ranking pipeline.

## Commands

```bash
python3 server.py            # run the ranking server (add --debug to log requests)
python genome.py             # GA demo: seeds from Samples/, breeds one child -> offspring_demo.svg
pip install -r requirements.txt   # only needed for analyze.ipynb (jupyter/numpy/pandas/matplotlib)
jupyter notebook analyze.ipynb    # Bradley-Terry ranking + bias analysis; Kernel -> Restart & Run All
```

Both `server.py` and `genome.py` are pure standard library — no install needed to run
either. `Samples/` and `votes.jsonl` are gitignored; they hold real vote/image data,
not code.

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
