"""PCA "eigenshape" space over the Maker's Mark genomes.

``genome.py``'s crossover treats every node/handle as an independent gene, which
scrambles the correlations *between* points that make the mark read as one
drawing.  This module fixes that by learning the axes of variation from the seed
population itself:

1. Every genome flattens to a fixed-length feature vector (``flatten``): node
   coordinates, free-handle vectors, and each interior tangent as an *angle*
   plus its two handle lengths.  Same topology in every sample, so the layout
   is identical for all genomes (``feature_layout``).
2. ``PCABasis.fit`` computes the mean shape and the principal components of the
   seed vectors.  With n seeds the covariance has rank <= n-1, so we take the
   Gram-matrix shortcut: eigendecompose the small n x n matrix ``X @ X.T``
   (pure-stdlib cyclic Jacobi, ``jacobi_eigh``) and map its eigenvectors back
   to feature space.
3. A genome in the GA is then just a coefficient vector in that basis.  Each
   coefficient is a whole-shape deformation direction observed in the real
   samples, so crossover/mutation on coefficients can only move points
   *together*, the way they move in actual marks -- incoherent shapes are
   nearly unrepresentable.

Angles are unwrapped against a reference before PCA (so the space is linear)
and scaled into px-comparable units by the mean handle length at their node (a
rotation of ``d_theta`` moves the handle tips ~ ``len * d_theta`` px).  Decode
is wrap-immune because it goes back through cos/sin.

Like ``genome.py``, pure standard library.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from genome import (CANVAS_H, CANVAS_W, MIN_HANDLE, PATH_ORDER, SEGMENTS,
                    Genome, PathGene, _blx, load_samples)

LAYOUT_VERSION = 1
NODE_MARGIN = 0.0                       # seed nodes reach the canvas edge
                                        # (x up to ~107.7, y up to ~123.1), so
                                        # clamp to the full canvas, not the
                                        # margin genome.mutate uses
ANGLE_W_MIN, ANGLE_W_MAX = 4.0, 60.0    # clamp for the px-per-radian weights
TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Feature layout: Genome <-> flat vector
# ---------------------------------------------------------------------------

def feature_layout() -> List[dict]:
    """One entry per scalar feature, in flatten() order.

    Per path (in ``PATH_ORDER``): node x,y pairs; start/end free handle dx,dy;
    then per interior tangent: angle theta, len_in, len_out.  Deterministic
    from the fixed topology, so two genomes always align element-wise.
    """
    layout: List[dict] = []
    for pid in PATH_ORDER:
        n_nodes = SEGMENTS[pid] + 1
        for k in range(n_nodes):
            layout.append({"path": pid, "kind": "node", "index": k, "axis": "x"})
            layout.append({"path": pid, "kind": "node", "index": k, "axis": "y"})
        for k in (0, 1):                              # 0 = start, 1 = end handle
            layout.append({"path": pid, "kind": "free", "index": k, "axis": "x"})
            layout.append({"path": pid, "kind": "free", "index": k, "axis": "y"})
        for t in range(n_nodes - 2):
            layout.append({"path": pid, "kind": "angle", "index": t, "axis": None})
            layout.append({"path": pid, "kind": "len_in", "index": t, "axis": None})
            layout.append({"path": pid, "kind": "len_out", "index": t, "axis": None})
    return layout


def angle_slots(layout: Sequence[dict]) -> List[int]:
    return [j for j, e in enumerate(layout) if e["kind"] == "angle"]


def flatten(g: Genome, ref_angles: Sequence[float] | None = None) -> List[float]:
    """Genome -> flat feature vector (see :func:`feature_layout`).

    ``ref_angles`` (one per tangent slot, flatten order) unwraps each angle by
    a multiple of 2*pi so it lands within pi of the reference -- required for
    angles to average/blend linearly across a population.
    """
    vec: List[float] = []
    ai = 0
    for pid in PATH_ORDER:
        pg = g.paths[pid]
        for (x, y) in pg.nodes:
            vec += [x, y]
        vec += [pg.start_handle[0], pg.start_handle[1],
                pg.end_handle[0], pg.end_handle[1]]
        for (ux, uy, li, lo) in pg.tangents:
            theta = math.atan2(uy, ux)
            if ref_angles is not None:
                theta += TWO_PI * round((ref_angles[ai] - theta) / TWO_PI)
            ai += 1
            vec += [theta, li, lo]
    return vec


def unflatten(vec: Sequence[float]) -> Genome:
    """Flat feature vector -> Genome, clamping geometry back to sanity.

    Nodes are clamped to the canvas and handle lengths floored at
    ``MIN_HANDLE`` -- BLX/Gaussian moves in coefficient space can push either
    out of range.
    """
    paths: Dict[str, PathGene] = {}
    i = 0
    for pid in PATH_ORDER:
        n_nodes = SEGMENTS[pid] + 1
        nodes = []
        for _ in range(n_nodes):
            x = min(max(vec[i], NODE_MARGIN), CANVAS_W - NODE_MARGIN)
            y = min(max(vec[i + 1], NODE_MARGIN), CANVAS_H - NODE_MARGIN)
            nodes.append((x, y))
            i += 2
        start_handle = (vec[i], vec[i + 1])
        end_handle = (vec[i + 2], vec[i + 3])
        i += 4
        tangents = []
        for _ in range(n_nodes - 2):
            theta, li, lo = vec[i], vec[i + 1], vec[i + 2]
            i += 3
            tangents.append((math.cos(theta), math.sin(theta),
                             max(MIN_HANDLE, li), max(MIN_HANDLE, lo)))
        paths[pid] = PathGene(nodes, tangents, start_handle, end_handle)
    return Genome(paths, {"origin": "eigen"})


# ---------------------------------------------------------------------------
# Cyclic Jacobi eigensolver (symmetric matrices; n ~ 22 here)
# ---------------------------------------------------------------------------

def jacobi_eigh(A: Sequence[Sequence[float]], tol: float = 1e-12,
                max_sweeps: int = 100) -> Tuple[List[float], List[List[float]]]:
    """Eigenvalues/vectors of a real symmetric matrix by cyclic Jacobi.

    Returns ``(eigenvalues, eigenvectors)`` sorted by descending eigenvalue;
    ``eigenvectors[i]`` is the (unit) eigenvector for ``eigenvalues[i]``.
    """
    n = len(A)
    a = [list(row) for row in A]
    v = [[1.0 if r == c else 0.0 for c in range(n)] for r in range(n)]
    scale = math.sqrt(sum(a[r][c] ** 2 for r in range(n) for c in range(n))) or 1.0

    for _ in range(max_sweeps):
        off = math.sqrt(sum(a[p][q] ** 2 for p in range(n) for q in range(p + 1, n)))
        if off <= tol * scale:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) <= 1e-300:
                    continue
                tau = (a[q][q] - a[p][p]) / (2.0 * apq)
                t = (1.0 if tau >= 0 else -1.0) / (abs(tau) + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                for r in range(n):                     # A <- Jt A J, columns p,q
                    arp, arq = a[r][p], a[r][q]
                    a[r][p] = c * arp - s * arq
                    a[r][q] = s * arp + c * arq
                for r in range(n):                     # rows p,q
                    apr, aqr = a[p][r], a[q][r]
                    a[p][r] = c * apr - s * aqr
                    a[q][r] = s * apr + c * aqr
                for r in range(n):                     # accumulate eigenvectors
                    vrp, vrq = v[r][p], v[r][q]
                    v[r][p] = c * vrp - s * vrq
                    v[r][q] = s * vrp + c * vrq

    evals = [a[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: evals[i], reverse=True)
    return ([evals[i] for i in order],
            [[v[r][i] for r in range(n)] for i in order])


# ---------------------------------------------------------------------------
# The basis
# ---------------------------------------------------------------------------

@dataclass
class PCABasis:
    """Mean shape + principal components fitted to a seed population.

    Everything needed to encode/decode is stored here (including the feature
    layout and angle weights), and round-trips through ``to_dict`` /
    ``from_dict`` -- a persisted run decodes identically forever, no matter
    what happens to ``Samples/`` afterwards.
    """

    layout: List[dict]
    weights: List[float]                  # per-feature scale, applied before PCA
    mean: List[float]                     # in *scaled* feature space
    components: List[List[float]]         # k rows, unit-norm, scaled space
    eigenvalues: List[float]              # Gram-matrix eigenvalues, descending
    stds: List[float]                     # sqrt(eigenvalue / (n-1)), per component
    n_seeds: int
    seed_files: List[str] = field(default_factory=list)
    layout_version: int = LAYOUT_VERSION

    @property
    def dim(self) -> int:
        return len(self.mean)

    @property
    def n_components(self) -> int:
        return len(self.components)

    # --- fitting ------------------------------------------------------------
    @classmethod
    def fit(cls, genomes: Sequence[Genome], var_keep: float = 1.0,
            eps: float = 1e-10) -> "PCABasis":
        n = len(genomes)
        if n < 2:
            raise ValueError(f"need at least 2 seed genomes, got {n}")
        layout = feature_layout()
        d = len(layout)
        aslots = angle_slots(layout)

        # Flatten with all angles unwrapped against the first seed's angles.
        f0 = flatten(genomes[0])
        ref_angles = [f0[j] for j in aslots]
        F = [flatten(g, ref_angles) for g in genomes]

        # Angle weights: mean handle length at that node (len slots follow the
        # angle slot immediately in the layout), clamped to a sane px range.
        weights = [1.0] * d
        for j in aslots:
            mean_len = sum((F[i][j + 1] + F[i][j + 2]) / 2.0 for i in range(n)) / n
            weights[j] = min(max(mean_len, ANGLE_W_MIN), ANGLE_W_MAX)

        S = [[weights[j] * F[i][j] for j in range(d)] for i in range(n)]
        mean = [sum(S[i][j] for i in range(n)) / n for j in range(d)]
        X = [[S[i][j] - mean[j] for j in range(d)] for i in range(n)]

        # Gram-matrix trick: eigendecompose the n x n X @ X.T instead of the
        # d x d covariance; identical non-zero spectrum, way smaller problem.
        G = [[sum(X[i][j] * X[k][j] for j in range(d)) for k in range(n)]
             for i in range(n)]
        evals, evecs = jacobi_eigh(G)

        total = sum(l for l in evals if l > 0.0) or 1.0
        floor = eps * max(evals[0], 0.0)
        components: List[List[float]] = []
        eigenvalues: List[float] = []
        cum = 0.0
        for lam, vec in zip(evals, evecs):
            if lam <= floor or lam <= 0.0:
                break
            norm = math.sqrt(lam)
            components.append([sum(X[i][j] * vec[i] for i in range(n)) / norm
                               for j in range(d)])
            eigenvalues.append(lam)
            cum += lam
            if var_keep < 1.0 and cum >= var_keep * total:
                break

        stds = [math.sqrt(lam / (n - 1)) for lam in eigenvalues]
        seed_files = [g.meta.get("origin", "") for g in genomes]
        return cls(layout, weights, mean, components, eigenvalues, stds,
                   n, seed_files)

    # --- encode / decode ------------------------------------------------------
    def _mean_angles(self) -> List[float]:
        return [self.mean[j] / self.weights[j] for j in angle_slots(self.layout)]

    def encode(self, g: Genome) -> List[float]:
        """Genome -> coefficient vector (length ``n_components``)."""
        f = flatten(g, ref_angles=self._mean_angles())
        x = [self.weights[j] * f[j] - self.mean[j] for j in range(self.dim)]
        return [sum(pc[j] * x[j] for j in range(self.dim)) for pc in self.components]

    def decode(self, coeffs: Sequence[float]) -> Genome:
        """Coefficient vector -> Genome."""
        s = list(self.mean)
        for c, pc in zip(coeffs, self.components):
            for j in range(self.dim):
                s[j] += c * pc[j]
        return unflatten([s[j] / self.weights[j] for j in range(self.dim)])

    # --- persistence ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "layout_version": self.layout_version,
            "layout": self.layout,
            "dim": self.dim,
            "weights": self.weights,
            "mean": self.mean,
            "components": self.components,
            "eigenvalues": self.eigenvalues,
            "stds": self.stds,
            "n_seeds": self.n_seeds,
            "seed_files": self.seed_files,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PCABasis":
        basis = cls(d["layout"], d["weights"], d["mean"], d["components"],
                    d["eigenvalues"], d["stds"], d["n_seeds"],
                    d.get("seed_files", []), d.get("layout_version", 0))
        if basis.layout_version != LAYOUT_VERSION or basis.dim != len(feature_layout()):
            raise ValueError(
                f"stored basis (layout v{basis.layout_version}, dim {basis.dim}) is "
                f"incompatible with this code (v{LAYOUT_VERSION}, dim {len(feature_layout())})"
            )
        return basis


# ---------------------------------------------------------------------------
# Breeding in coefficient space
# ---------------------------------------------------------------------------

def crossover_coeffs(a: Sequence[float], b: Sequence[float], *,
                     blend_prob: float = 0.5, alpha: float = 0.5,
                     rng: random.Random | None = None) -> List[float]:
    """Per-coefficient inherit-or-BLX, the same scheme genome.crossover uses.

    Because each coefficient is a whole-shape deformation direction, inheriting
    or blending it keeps every point moving coherently.
    """
    rng = rng or random.Random()
    out: List[float] = []
    for av, bv in zip(a, b):
        if rng.random() < blend_prob:
            out.append(_blx(av, bv, alpha, rng))
        else:
            out.append(av if rng.random() < 0.5 else bv)
    return out


def mutate_coeffs(coeffs: Sequence[float], stds: Sequence[float], *,
                  rate: float = 0.3, sigma: float = 0.35,
                  rng: random.Random | None = None) -> List[float]:
    """Gaussian jitter per coefficient, scaled by that component's population
    std-dev -- big along directions the seeds actually vary, negligible along
    near-constant ones."""
    rng = rng or random.Random()
    return [c + rng.gauss(0.0, sigma * s) if rng.random() < rate else c
            for c, s in zip(coeffs, stds)]


def breed_coeffs(a: Sequence[float], b: Sequence[float], stds: Sequence[float], *,
                 blend_prob: float = 0.5, alpha: float = 0.5,
                 rate: float = 0.3, sigma: float = 0.35,
                 rng: random.Random | None = None) -> List[float]:
    """Crossover then mutate: one offspring coefficient vector."""
    rng = rng or random.Random()
    child = crossover_coeffs(a, b, blend_prob=blend_prob, alpha=alpha, rng=rng)
    return mutate_coeffs(child, stds, rate=rate, sigma=sigma, rng=rng)


# ---------------------------------------------------------------------------
# Demo: fit the space, check round-trip, breed one child
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = random.Random(7)
    pop = load_samples()
    basis = PCABasis.fit(pop)
    total = sum(basis.eigenvalues)
    top3 = sum(basis.eigenvalues[:3]) / total * 100
    print(f"fitted {basis.n_components} components from {basis.n_seeds} seeds "
          f"({basis.dim} features); top 3 explain {top3:.1f}% of variance")

    err = max(
        abs(u - w)
        for g in pop
        for u, w in zip(flatten(basis.decode(basis.encode(g)),
                                basis._mean_angles()),
                        flatten(g, basis._mean_angles()))
    )
    print(f"max round-trip feature error across all seeds: {err:.2e}")

    ca, cb = basis.encode(pop[3]), basis.encode(pop[11])
    child = basis.decode(breed_coeffs(ca, cb, basis.stds, rng=rng))
    child.save_svg("eigen_demo.svg")
    print(f"bred {pop[3].meta['origin']} x {pop[11].meta['origin']} -> eigen_demo.svg")
