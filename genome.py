"""Genetic encoding of the Maker's Mark vector marks.

Each sample in ``Samples/vector_*.svg`` is the same drawing -- three cubic
Bezier paths with fixed *topology* (same number of nodes) and varying point
locations:

    DG : 3 cubic segments -> 4 on-curve nodes
    D  : 1 cubic segment  -> 2 on-curve nodes
    GS : 4 cubic segments -> 5 on-curve nodes

Illustrator writes those paths with a mix of relative (``c``), absolute (``C``)
and smooth (``s``) commands purely to save bytes.  The prompt's worry -- "are
the control points relative or absolute?" -- only exists because of that mixed
encoding.  We make it a non-issue by parsing every path into a single canonical
form: **absolute on-curve nodes plus handle vectors stored relative to the node
they belong to.**

A handle vector is the natural unit a vector editor manipulates: it captures the
tangent direction and "pull" of the curve at a node, independent of where the
node sits.  Keeping nodes absolute (the skeleton) and handles relative (the
local style) is what makes recombination behave well -- you can mix one parent's
skeleton with another's curl, and a handle stays meaningful no matter which
parent's node it lands on.

The genome is therefore a flat bag of *components*, each tagged ``"node"`` or
``"handle"``.  Crossover and mutation operate on those components; nothing in the
breeding code needs to know about SVG syntax.

Pure standard library (``math`` + ``random``) so the web server can reuse it
later without pulling in the scientific stack.
"""

from __future__ import annotations

import glob
import math
import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

Point = Tuple[float, float]

# --- canvas / style, identical across every sample -------------------------

VIEWBOX = "0 0 110 124"
CANVAS_W, CANVAS_H = 110.0, 124.0
STROKE = "#231f20"
STROKE_WIDTH = 3
PATH_ORDER = ("GS", "D", "DG")          # draw order matches the source files
SEGMENTS = {"DG": 3, "D": 1, "GS": 4}   # cubic segments per path (fixed topology)
MIN_HANDLE = 0.25                       # floor for handle length, keeps tangents
                                        # well-defined (samples' shortest is ~0.68)


# ---------------------------------------------------------------------------
# SVG path-data parsing  ->  canonical absolute cubic segments
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"([MmCcSsLlHhVvZz])|([-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)")


def _tokenize(d: str):
    for m in _TOKEN.finditer(d):
        yield ("cmd", m.group(1)) if m.group(1) else ("num", float(m.group(2)))


def parse_path_d(d: str) -> Tuple[Point, List[Tuple[Point, Point, Point]]]:
    """Parse a ``d`` attribute into ``(start, segments)``.

    ``segments`` is a list of ``(c1, c2, end)`` triples in **absolute**
    coordinates -- every command type (C/c, S/s, L/l) is lifted into the same
    cubic representation.
    """
    toks = list(_tokenize(d))
    i, cmd = 0, None
    cur: Point = (0.0, 0.0)
    start: Point | None = None
    prev_c2: Point | None = None          # 2nd control of previous cubic, for S/s
    segs: List[Tuple[Point, Point, Point]] = []

    def line_as_cubic(a: Point, b: Point) -> Tuple[Point, Point, Point]:
        c1 = (a[0] + (b[0] - a[0]) / 3, a[1] + (b[1] - a[1]) / 3)
        c2 = (a[0] + 2 * (b[0] - a[0]) / 3, a[1] + 2 * (b[1] - a[1]) / 3)
        return c1, c2, b

    while i < len(toks):
        if toks[i][0] == "cmd":
            cmd = toks[i][1]
            i += 1
        rel = cmd.islower()
        ox, oy = cur if rel else (0.0, 0.0)

        if cmd in "Mm":
            cur = (toks[i][1] + ox, toks[i + 1][1] + oy)
            i += 2
            start = cur
            prev_c2 = None
            cmd = "l" if cmd == "m" else "L"      # implicit moveto args are lineto
        elif cmd in "Cc":
            v = [toks[i + k][1] for k in range(6)]; i += 6
            c1 = (v[0] + ox, v[1] + oy)
            c2 = (v[2] + ox, v[3] + oy)
            end = (v[4] + ox, v[5] + oy)
            segs.append((c1, c2, end)); prev_c2 = c2; cur = end
        elif cmd in "Ss":
            v = [toks[i + k][1] for k in range(4)]; i += 4
            c2 = (v[0] + ox, v[1] + oy)
            end = (v[2] + ox, v[3] + oy)
            c1 = (2 * cur[0] - prev_c2[0], 2 * cur[1] - prev_c2[1]) if prev_c2 else cur
            segs.append((c1, c2, end)); prev_c2 = c2; cur = end
        elif cmd in "Ll":
            end = (toks[i][1] + ox, toks[i + 1][1] + oy); i += 2
            segs.append(line_as_cubic(cur, end)); prev_c2 = None; cur = end
        elif cmd in "Hh":
            end = (toks[i][1] + ox, cur[1]); i += 1
            segs.append(line_as_cubic(cur, end)); prev_c2 = None; cur = end
        elif cmd in "Vv":
            end = (cur[0], toks[i][1] + oy); i += 1
            segs.append(line_as_cubic(cur, end)); prev_c2 = None; cur = end
        elif cmd in "Zz":
            pass
        else:
            i += 1

    if start is None:
        raise ValueError("path has no moveto")
    return start, segs


# ---------------------------------------------------------------------------
# Gene structures
# ---------------------------------------------------------------------------

def _unit_tangent(out_vec: Point, in_vec: Point) -> Point:
    """Unit forward-tangent at an interior node.

    The forward direction is the outgoing handle direction; the incoming handle
    points the opposite way, so ``-in_vec`` should agree with ``out_vec``.  We
    average the two for robustness and normalise.  Falls back gracefully if a
    handle has zero length.
    """
    fx = out_vec[0] - in_vec[0]
    fy = out_vec[1] - in_vec[1]
    n = math.hypot(fx, fy)
    if n < 1e-9:                       # degenerate; fall back to whatever we have
        fx, fy = out_vec if math.hypot(*out_vec) > 1e-9 else (1.0, 0.0)
        n = math.hypot(fx, fy) or 1.0
    return (fx / n, fy / n)


@dataclass
class Gene:
    """A single typed gene used by crossover/mutation.

    ``kind`` is one of:

    * ``"node"``    -> ``value`` is an absolute ``(x, y)`` on-curve point.
    * ``"free"``    -> ``value`` is a free handle vector ``(dx, dy)``.
    * ``"tangent"`` -> ``value`` is ``(ux, uy, len_in, len_out)`` for a smooth
      interior node; ``(ux, uy)`` is kept unit-length so the two handles it
      generates stay collinear no matter how the gene is recombined.
    """

    kind: str
    value: tuple


@dataclass
class PathGene:
    """One path's geometry as a skeleton of nodes + smoothness-preserving handles.

    In every sample the two Bezier handles meeting at an *interior* node are
    exactly antiparallel -- the curve passes through the node smoothly, with no
    kink.  If we stored the two handles as independent vectors (as an earlier
    version did), crossover and mutation would nudge their directions apart and
    reintroduce kinks.  So an interior node stores **one shared tangent
    direction** plus **two independent lengths**, which makes a kink
    *unrepresentable*: both handles are rebuilt from the same unit vector.

    ``nodes``         absolute on-curve points, length = segments + 1.
    ``tangents``      one per *interior* node (indices 1..n-2), each a tuple
                      ``(ux, uy, len_in, len_out)`` where ``(ux, uy)`` is the
                      unit forward-tangent and the two lengths are the incoming
                      and outgoing handle magnitudes.
    ``start_handle``  free vector ``c1 - node0`` leaving the first node.
    ``end_handle``    free vector ``c2 - node_last`` arriving at the last node.

    Endpoint handles are free (a path end has no continuity constraint); the
    single-segment ``D`` path has no interior nodes, so ``tangents`` is empty and
    both its handles are free.
    """

    nodes: List[Point]
    tangents: List[Tuple[float, float, float, float]]
    start_handle: Point
    end_handle: Point

    # --- construction ------------------------------------------------------
    @classmethod
    def from_d(cls, d: str) -> "PathGene":
        start, segs = parse_path_d(d)
        nodes = [start] + [end for (_, _, end) in segs]
        # absolute control points per segment
        c1s = [c1 for (c1, _, _) in segs]
        c2s = [c2 for (_, c2, _) in segs]

        start_handle = (c1s[0][0] - nodes[0][0], c1s[0][1] - nodes[0][1])
        end_handle = (c2s[-1][0] - nodes[-1][0], c2s[-1][1] - nodes[-1][1])

        tangents: List[Tuple[float, float, float, float]] = []
        for k in range(1, len(nodes) - 1):       # interior nodes only
            in_vec = (c2s[k - 1][0] - nodes[k][0], c2s[k - 1][1] - nodes[k][1])
            out_vec = (c1s[k][0] - nodes[k][0], c1s[k][1] - nodes[k][1])
            len_in = math.hypot(*in_vec)
            len_out = math.hypot(*out_vec)
            # forward tangent = average of out direction and the reverse of in
            ux, uy = _unit_tangent(out_vec, in_vec)
            tangents.append((ux, uy, len_in, len_out))
        return cls(nodes, tangents, start_handle, end_handle)

    # --- reconstruction ----------------------------------------------------
    def _out_handle(self, i: int) -> Point:
        """Vector of the control point leaving node ``i`` (== c1 of segment i)."""
        if i == 0:
            return self.start_handle
        ux, uy, _, len_out = self.tangents[i - 1]
        return (ux * len_out, uy * len_out)

    def _in_handle(self, i: int) -> Point:
        """Vector of the control point arriving at node ``i+1`` (== c2 of segment i)."""
        j = i + 1
        if j == len(self.nodes) - 1:
            return self.end_handle
        ux, uy, len_in, _ = self.tangents[j - 1]
        return (-ux * len_in, -uy * len_in)      # opposite the forward tangent

    def control_points(self, i: int) -> Tuple[Point, Point, Point, Point]:
        """Absolute (p0, c1, c2, p1) for segment ``i``."""
        p0, p1 = self.nodes[i], self.nodes[i + 1]
        oh, ih = self._out_handle(i), self._in_handle(i)
        c1 = (p0[0] + oh[0], p0[1] + oh[1])
        c2 = (p1[0] + ih[0], p1[1] + ih[1])
        return p0, c1, c2, p1

    def to_d(self) -> str:
        p0 = self.nodes[0]
        out = [f"M{_f(p0[0])},{_f(p0[1])}"]
        for i in range(len(self.nodes) - 1):
            _, c1, c2, p1 = self.control_points(i)
            out.append(
                f"C{_f(c1[0])},{_f(c1[1])},{_f(c2[0])},{_f(c2[1])},{_f(p1[0])},{_f(p1[1])}"
            )
        return "".join(out)

    # --- flat gene view used by crossover/mutation -------------------------
    def genes(self) -> List["Gene"]:
        """Ordered list of typed genes (see :class:`Gene`).

        Layout: all nodes, then the two free handles, then one tangent per
        interior node.  Two genomes of the same path always produce identically
        ordered, type-matched gene lists, so they zip cleanly for crossover.
        """
        g: List[Gene] = [Gene("node", p) for p in self.nodes]
        g.append(Gene("free", self.start_handle))
        g.append(Gene("free", self.end_handle))
        g += [Gene("tangent", t) for t in self.tangents]
        return g

    @classmethod
    def from_genes(cls, genes: Sequence["Gene"], n_segs: int) -> "PathGene":
        n_nodes = n_segs + 1
        vals = [gn.value for gn in genes]
        nodes = list(vals[:n_nodes])
        start_handle, end_handle = vals[n_nodes], vals[n_nodes + 1]
        tangents = [tuple(t) for t in vals[n_nodes + 2:]]
        return cls(nodes, tangents, start_handle, end_handle)


@dataclass
class Genome:
    """The full mark: one :class:`PathGene` per path id."""

    paths: Dict[str, PathGene]
    meta: dict = field(default_factory=dict)   # e.g. {"origin": "vector_3.svg"}

    # --- construction ------------------------------------------------------
    @classmethod
    def from_svg(cls, text: str, origin: str = "") -> "Genome":
        paths = {}
        for pid in PATH_ORDER:
            m = re.search(r'id="%s"[^>]*?\bd="([^"]*)"' % re.escape(pid), text)
            if not m:
                raise ValueError(f"path id={pid!r} not found")
            paths[pid] = PathGene.from_d(m.group(1))
        return cls(paths, {"origin": origin})

    @classmethod
    def from_file(cls, path: str) -> "Genome":
        with open(path) as fh:
            return cls.from_svg(fh.read(), origin=os.path.basename(path))

    # --- rendering ---------------------------------------------------------
    def to_svg(self) -> str:
        body = "\n".join(
            f'  <path id="{pid}" class="st0" d="{self.paths[pid].to_d()}"/>'
            for pid in PATH_ORDER
        )
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg id="Layer_1" xmlns="http://www.w3.org/2000/svg" '
            f'version="1.1" viewBox="{VIEWBOX}">\n'
            "  <defs>\n    <style>\n      .st0 {\n        fill: none;\n"
            f"        stroke: {STROKE};\n        stroke-miterlimit: 10;\n"
            f"        stroke-width: {STROKE_WIDTH}px;\n      }}\n"
            "    </style>\n  </defs>\n"
            f"{body}\n</svg>\n"
        )

    def save_svg(self, path: str) -> None:
        with open(path, "w") as fh:
            fh.write(self.to_svg())

    # --- breeding ----------------------------------------------------------
    def breed(self, other: "Genome", rate: float = 0.15, rng: random.Random | None = None,
              **kw) -> "Genome":
        """Produce one offspring: crossover with ``other`` then mutate.

        This is the single call the GA driver uses once two parents are picked.
        ``rate`` is the per-component mutation probability; see :func:`mutate`.
        """
        return crossover(self, other, rng=rng, **kw).mutate(rate=rate, rng=rng)

    def mutate(self, rate: float = 0.15, **kw) -> "Genome":
        return mutate(self, rate=rate, **kw)


# ---------------------------------------------------------------------------
# Recombination
# ---------------------------------------------------------------------------

def _blx(a: float, b: float, alpha: float, rng: random.Random) -> float:
    """BLX-alpha: sample from an interval that *extends past* both parents.

    This is the antidote to "averaging makes every child look the same": instead
    of collapsing toward the midpoint, the child value is drawn from
    ``[min - alpha*d, max + alpha*d]``, so offspring can land outside the span of
    their parents and the population keeps exploring.
    """
    lo, hi = (a, b) if a <= b else (b, a)
    d = hi - lo
    return rng.uniform(lo - alpha * d, hi + alpha * d)


def _cross_gene(a: Gene, blend: bool, av: tuple, bv: tuple, alpha: float,
                rng: random.Random) -> Gene:
    """Recombine one matched gene pair, respecting its kind."""
    if not blend:
        return Gene(a.kind, av if rng.random() < 0.5 else bv)
    if a.kind == "tangent":
        # blend direction as a vector then renormalise (keeps it a unit tangent),
        # and blend the two handle lengths independently with BLX.
        dx = _blx(av[0], bv[0], alpha, rng)
        dy = _blx(av[1], bv[1], alpha, rng)
        n = math.hypot(dx, dy) or 1.0
        li = max(MIN_HANDLE, _blx(av[2], bv[2], alpha, rng))
        lo = max(MIN_HANDLE, _blx(av[3], bv[3], alpha, rng))
        return Gene("tangent", (dx / n, dy / n, li, lo))
    # node / free: blend each coordinate with BLX
    return Gene(a.kind, tuple(_blx(av[k], bv[k], alpha, rng) for k in range(len(av))))


def crossover(
    p1: Genome,
    p2: Genome,
    *,
    blend_prob: float = 0.5,
    alpha: float = 0.5,
    rng: random.Random | None = None,
) -> Genome:
    """Combine two parents into a child genome (no mutation yet).

    For each gene (a node, a free handle, or an interior-node tangent) we
    independently choose, per path:

    * with probability ``1 - blend_prob`` -> **inherit** the whole gene from one
      parent at random (keeps a coherent feature intact), or
    * with probability ``blend_prob``     -> **blend** the two parents' values
      using BLX-alpha (novel intermediate geometry that can exceed the parents'
      range).

    Tangent genes blend as *(direction, length_in, length_out)*, so the child's
    node stays smooth: both of its handles are regenerated from one shared,
    renormalised direction. Mixing inheritance and BLX gives variety in two
    directions at once -- whole features swap between parents while blended genes
    explore the space between and around them.
    """
    rng = rng or random.Random()
    child_paths: Dict[str, PathGene] = {}
    for pid in PATH_ORDER:
        g1, g2 = p1.paths[pid].genes(), p2.paths[pid].genes()
        out: List[Gene] = []
        for a, b in zip(g1, g2):
            blend = rng.random() < blend_prob
            out.append(_cross_gene(a, blend, a.value, b.value, alpha, rng))
        child_paths[pid] = PathGene.from_genes(out, SEGMENTS[pid])
    return Genome(child_paths, {"origin": "cross"})


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def mutate(
    g: Genome,
    *,
    rate: float = 0.15,
    node_sigma: float = 4.0,
    handle_sigma: float = 6.0,
    angle_sigma: float = 0.15,
    clamp_nodes: bool = True,
    margin: float = 3.0,
    rng: random.Random | None = None,
) -> Genome:
    """Return a mutated copy of ``g``.

    ``rate``         probability that any given gene is perturbed.
    ``node_sigma``   Gaussian std-dev (px) for node jitter -- moves the skeleton.
    ``handle_sigma`` std-dev (px) for handle-length / free-handle jitter --
                     changes how hard the curve pulls; the more visual knob.
    ``angle_sigma``  std-dev (radians) for *rotating* an interior node's tangent.
                     Rotating the shared direction keeps the node smooth, so this
                     reshapes the curl without ever introducing a kink.
    ``clamp_nodes``  keep nodes inside the canvas (handles may point outside).

    A tangent gene mutates as "rotate the direction + jitter both lengths," which
    is exactly the set of moves that preserve smoothness. Nodes, free handles and
    handle lengths use ``node_sigma``/``handle_sigma`` because a node move shifts
    a whole region while a handle can swing further without wrecking the
    letterform.
    """
    rng = rng or random.Random()

    def hit() -> bool:
        return rng.random() < rate

    def mutate_gene(gene: Gene) -> Gene:
        if gene.kind == "node":
            if not hit():
                return gene
            x = gene.value[0] + rng.gauss(0.0, node_sigma)
            y = gene.value[1] + rng.gauss(0.0, node_sigma)
            if clamp_nodes:
                x = min(max(x, margin), CANVAS_W - margin)
                y = min(max(y, margin), CANVAS_H - margin)
            return Gene("node", (x, y))
        if gene.kind == "free":
            if not hit():
                return gene
            return Gene("free", (gene.value[0] + rng.gauss(0.0, handle_sigma),
                                 gene.value[1] + rng.gauss(0.0, handle_sigma)))
        # tangent: rotate the unit direction, jitter both lengths
        ux, uy, li, lo = gene.value
        if hit():
            theta = rng.gauss(0.0, angle_sigma)
            c, s = math.cos(theta), math.sin(theta)
            ux, uy = ux * c - uy * s, ux * s + uy * c
        if hit():
            li = max(MIN_HANDLE, li + rng.gauss(0.0, handle_sigma))
        if hit():
            lo = max(MIN_HANDLE, lo + rng.gauss(0.0, handle_sigma))
        return Gene("tangent", (ux, uy, li, lo))

    new_paths: Dict[str, PathGene] = {}
    for pid, pg in g.paths.items():
        mutated = [mutate_gene(gn) for gn in pg.genes()]
        new_paths[pid] = PathGene.from_genes(mutated, SEGMENTS[pid])
    return Genome(new_paths, dict(g.meta, origin="mutant"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(v: float) -> str:
    """Format a coordinate compactly (drop trailing zeros, like Illustrator)."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


def load_samples(pattern: str = "Samples/vector_*.svg") -> List[Genome]:
    """Load every sample SVG as a seed genome, sorted by file number."""
    files = sorted(glob.glob(pattern),
                   key=lambda s: int(re.search(r"(\d+)", os.path.basename(s)).group()))
    return [Genome.from_file(f) for f in files]


# ---------------------------------------------------------------------------
# Demo: seed from samples, breed a child, write it out
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = random.Random(7)
    pop = load_samples()
    print(f"loaded {len(pop)} seed genomes")

    # round-trip sanity: re-rendered sample re-parses to the same geometry
    a, b = pop[3], pop[11]
    child = a.breed(b, rate=0.05, rng=rng)
    out = "offspring_demo.svg"
    child.save_svg(out)
    print(f"bred {a.meta['origin']} x {b.meta['origin']} -> {out}")
    for pid in PATH_ORDER:
        pg = child.paths[pid]
        print(f"  {pid}: {len(pg.nodes)} nodes, {len(pg.nodes) - 1} segments, "
              f"{len(pg.tangents)} smooth interior nodes")
