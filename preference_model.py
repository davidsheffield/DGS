"""Bayesian *peaked* preference model over the eigenshape space.

The evolver (``evolve_server.py``) asks you to pick which grid cells to breed;
this instead learns, from forced A/B duels, **what values along each eigen-axis
you prefer** -- and, crucially, that the preference has a *peak*: being on either
side of a sweet spot is worse, so votes can drift back and forth near the
optimum without a clear winner.

Model
-----
A candidate is a coefficient vector ``c`` in an ``eigen.PCABasis`` (length K =
``basis.n_components``).  Standardize by the per-component population std so a
unit is comparable across axes::

    z_k = c_k / stds_k

and use a **per-axis quadratic** feature map::

    phi(c) = [z_1..z_K, z_1^2 .. z_K^2]            (length 2K)

with utility ``U(c) = w . phi(c)`` and preference
``P(a > b) = sigmoid(w . (phi(a) - phi(b)))`` (the constant cancels in the
difference, so there is no intercept).  The ``z_k^2`` terms give each axis an
interior optimum at ``z* = -w_lin / (2 w_quad)`` when ``w_quad < 0`` -- a real
peak; ``w_quad >= 0`` means that axis has no interior peak (an *edge*
preference), which is representable and reported as such.

Posterior
---------
Bayesian logistic regression with a Gaussian prior ``w ~ N(0, prior_var * I)``,
fitted by Newton/IRLS on the per-vote difference vectors; the Laplace
approximation ``N(w_MAP, H^-1)`` (H = Hessian at the mode) is used for Thompson
sampling.  Standardized features keep everything O(1) and well-conditioned, and
2K ~ 40, so a pure-Python Cholesky solve is ample.

Active sampling
---------------
``next_duel`` is dueling Thompson sampling: draw two independent weight vectors
from the posterior and show the argmax candidate of each.  A diffuse posterior
(early) gives varied, exploratory duels; a sharp one (later) makes both land
near the optimum, so similar shapes recur -- the intended back-and-forth.  A
recency buffer forbids showing anything close to the last few marks, so a shape
only returns after a palette cleanser.

Pure standard library, like ``genome.py`` / ``eigen.py``.
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import List, Sequence, Tuple

WINNER_Y = {"a": 1.0, "b": 0.0, "tie": 0.5}


# ---------------------------------------------------------------------------
# Small linear-algebra helpers (symmetric positive-definite; D ~ 40)
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _softplus(x: float) -> float:
    """log(1 + e^x), overflow-safe."""
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def _cholesky(A: List[List[float]]) -> List[List[float]]:
    """Lower-triangular L with A = L Lᵀ (A SPD; the prior term guarantees it)."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Li = L[i]
        for j in range(i + 1):
            Lj = L[j]
            s = sum(Li[k] * Lj[k] for k in range(j))
            if i == j:
                Li[j] = math.sqrt(max(A[i][i] - s, 1e-12))
            else:
                Li[j] = (A[i][j] - s) / Lj[j]
    return L


def _forward(L: List[List[float]], b: Sequence[float]) -> List[float]:
    """Solve L y = b (L lower-triangular)."""
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        Li = L[i]
        y[i] = (b[i] - sum(Li[k] * y[k] for k in range(i))) / Li[i]
    return y


def _backward_LT(L: List[List[float]], b: Sequence[float]) -> List[float]:
    """Solve Lᵀ x = b (L lower-triangular, so Lᵀ upper-triangular)."""
    n = len(L)
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (b[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
    return x


def _chol_solve(L: List[List[float]], b: Sequence[float]) -> List[float]:
    """Solve (L Lᵀ) x = b."""
    return _backward_LT(L, _forward(L, b))


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class PreferenceModel:
    def __init__(self, stds: Sequence[float], *, prior_var: float = 1.0,
                 z_max: float = 2.5, pool_size: int = 200,
                 recent: int = 12, d_min: float = 1.5,
                 rng: random.Random | None = None):
        # Guard near-constant axes so z = c/std stays finite.
        self.stds = [s if abs(s) > 1e-9 else 1e-9 for s in stds]
        self.K = len(self.stds)
        self.D = 2 * self.K
        self.prior_var = prior_var
        self.z_max = z_max
        self.pool_size = pool_size
        self.d_min = d_min
        self.rng = rng or random.Random()
        self._obs: List[Tuple[List[float], float]] = []   # (diff vector, y)
        self._recent: "deque[List[float]]" = deque(maxlen=recent)
        self.w = [0.0] * self.D
        self._refit()

    @property
    def n_obs(self) -> int:
        return len(self._obs)

    # --- feature map --------------------------------------------------------
    def _z(self, coeffs: Sequence[float]) -> List[float]:
        return [c / s for c, s in zip(coeffs, self.stds)]

    def phi(self, coeffs: Sequence[float]) -> List[float]:
        z = self._z(coeffs)
        return z + [zi * zi for zi in z]

    def utility(self, coeffs: Sequence[float],
                w: Sequence[float] | None = None) -> float:
        w = self.w if w is None else w
        ph = self.phi(coeffs)
        return sum(w[j] * ph[j] for j in range(self.D))

    # --- observing votes ----------------------------------------------------
    def _diff(self, a_coeffs, b_coeffs) -> List[float]:
        pa, pb = self.phi(a_coeffs), self.phi(b_coeffs)
        return [pa[j] - pb[j] for j in range(self.D)]

    def observe(self, a_coeffs, b_coeffs, winner: str) -> None:
        self._obs.append((self._diff(a_coeffs, b_coeffs), WINNER_Y[winner]))
        self._refit()

    def observe_many(self, votes: Sequence[dict]) -> None:
        """Bulk-load logged votes (each ``{a_coeffs, b_coeffs, winner}``) then
        refit once -- used to warm-start from ``votes.jsonl`` at startup."""
        for v in votes:
            self._obs.append((self._diff(v["a_coeffs"], v["b_coeffs"]),
                              WINNER_Y[v["winner"]]))
        self._refit()

    # --- posterior (Laplace / IRLS) -----------------------------------------
    def _neg_log_post(self, w: Sequence[float]) -> float:
        """Penalized negative log-likelihood (the convex objective IRLS minimizes)."""
        lam = 1.0 / self.prior_var
        val = 0.5 * lam * sum(wj * wj for wj in w)
        for d, y in self._obs:
            s = sum(w[j] * d[j] for j in range(self.D))
            val += _softplus(s) - y * s          # cross-entropy, stable form
        return val

    def _refit(self, max_iter: int = 50) -> None:
        D = self.D
        lam = 1.0 / self.prior_var               # prior precision
        w = list(self.w)                         # warm-start from last fit
        L = None
        f = self._neg_log_post(w)
        for _ in range(max_iter):
            g = [lam * w[j] for j in range(D)]
            H = [[lam if i == j else 0.0 for j in range(D)] for i in range(D)]
            for d, y in self._obs:
                s = sum(w[j] * d[j] for j in range(D))
                p = _sigmoid(s)
                r = p - y
                wr = max(p * (1.0 - p), 1e-9)
                for i in range(D):
                    di = d[i]
                    if di == 0.0:
                        continue
                    g[i] += r * di
                    Hi = H[i]
                    wd = wr * di
                    for j in range(i, D):
                        Hi[j] += wd * d[j]
            for i in range(D):                   # symmetrize upper -> lower
                for j in range(i + 1, D):
                    H[j][i] = H[i][j]
            L = _cholesky(H)
            step = _chol_solve(L, g)             # Newton direction H^-1 g
            decrement = sum(g[j] * step[j] for j in range(D))   # = gᵀH^-1g >= 0
            if decrement < 1e-10:
                break
            t = 1.0                              # Armijo backtracking line search
            while True:
                w_new = [w[j] - t * step[j] for j in range(D)]
                f_new = self._neg_log_post(w_new)
                if f_new <= f - 1e-4 * t * decrement or t < 1e-6:
                    break
                t *= 0.5
            w, f = w_new, f_new
        self.w = w
        self._L = L if L is not None else _cholesky(
            [[lam if i == j else 0.0 for j in range(D)] for i in range(D)])

    # --- posterior samples / uncertainty ------------------------------------
    def sample_w(self, rng: random.Random | None = None) -> List[float]:
        """Draw w ~ N(w_MAP, H^-1).  If Lᵀu = n then Cov(u) = L^-ᵀL^-1 = H^-1."""
        rng = rng or self.rng
        n = [rng.gauss(0.0, 1.0) for _ in range(self.D)]
        u = _backward_LT(self._L, n)
        return [self.w[j] + u[j] for j in range(self.D)]

    def weight_std(self, j: int) -> float:
        """Posterior std of weight j: (H^-1)_jj = ||L^-1 e_j||²."""
        e = [1.0 if i == j else 0.0 for i in range(self.D)]
        m = _forward(self._L, e)
        return math.sqrt(sum(v * v for v in m))

    # --- what's preferred ---------------------------------------------------
    def preferred_z(self) -> List[Tuple[float, bool]]:
        """Per axis: (preferred standardized value z*, is_interior_peak)."""
        out: List[Tuple[float, bool]] = []
        for k in range(self.K):
            b = self.w[k]                    # linear weight
            a = self.w[self.K + k]           # quadratic weight
            if a < -1e-9:                    # concave -> interior peak
                z = max(-self.z_max, min(self.z_max, -b / (2.0 * a)))
                out.append((z, True))
            else:                            # convex/linear -> best endpoint
                zp, zm = self.z_max, -self.z_max
                z = zp if (b * zp + a * zp * zp) >= (b * zm + a * zm * zm) else zm
                out.append((z, False))
        return out

    def preferred_coeffs(self) -> List[float]:
        return [z * self.stds[k] for k, (z, _) in enumerate(self.preferred_z())]

    def axis_report(self) -> List[dict]:
        """One dict per axis for the analysis view, sorted by how strongly the
        votes constrain it (utility span across the axis)."""
        pref = self.preferred_z()
        rows = []
        for k in range(self.K):
            b = self.w[k]
            a = self.w[self.K + k]
            us = [b * z + a * z * z
                  for z in (i / 20.0 * self.z_max for i in range(-20, 21))]
            rows.append({
                "axis": k,
                "z_star": pref[k][0],
                "peak": pref[k][1],
                "lin": b, "quad": a,
                "lin_std": self.weight_std(k),
                "quad_std": self.weight_std(self.K + k),
                "span": max(us) - min(us),
            })
        rows.sort(key=lambda r: r["span"], reverse=True)
        return rows

    # --- active next-duel (dueling Thompson sampling) -----------------------
    def _zdist(self, c1: Sequence[float], c2: Sequence[float]) -> float:
        return math.sqrt(sum(((x - y) / s) ** 2
                             for x, y, s in zip(c1, c2, self.stds)))

    def _too_recent(self, coeffs: Sequence[float]) -> bool:
        return any(self._zdist(coeffs, r) < self.d_min for r in self._recent)

    def _candidate_pool(self, rng: random.Random) -> List[List[float]]:
        opt = self.preferred_coeffs()
        opt_active = any(abs(o) > 1e-9 for o in opt)
        pool: List[List[float]] = []
        for _ in range(self.pool_size * 12):
            if len(pool) >= self.pool_size:
                break
            if opt_active and pool and rng.random() < 0.25:
                # revisit the neighbourhood of the current optimum
                z = [opt[k] / self.stds[k] + rng.gauss(0.0, 0.6)
                     for k in range(self.K)]
            else:
                z = [rng.gauss(0.0, 1.0) for _ in range(self.K)]
            z = [max(-self.z_max, min(self.z_max, zi)) for zi in z]
            c = [z[k] * self.stds[k] for k in range(self.K)]
            if self._too_recent(c):
                continue
            pool.append(c)
        if not pool:                         # recency starved the pool: relax it
            z = [max(-self.z_max, min(self.z_max, rng.gauss(0.0, 1.0)))
                 for _ in range(self.K)]
            pool.append([z[k] * self.stds[k] for k in range(self.K)])
        return pool

    def _argmax(self, pool, w, *, avoid=None, min_dist: float = 0.0):
        best = None
        best_u = None
        for c in pool:
            if avoid is not None and self._zdist(c, avoid) < min_dist:
                continue
            u = self.utility(c, w)
            if best_u is None or u > best_u:
                best_u, best = u, c
        return best

    def explore_prob(self) -> float:
        """Fraction of duels that should be exploratory, annealed with data.

        While the posterior is flat a Thompson-argmax lands on the corners of
        the sampling box and never reveals *where* the interior peak is, so the
        curvature (``w_quad``) can't be learned.  Early duels therefore lean
        exploratory (random interior pairs); once curvature is known, Thompson
        exploitation correctly settles near the peak on its own.
        """
        return max(0.25, math.exp(-self.n_obs / 50.0))

    def next_duel(self, rng: random.Random | None = None
                  ) -> Tuple[List[float], List[float]]:
        rng = rng or self.rng
        pool = self._candidate_pool(rng)
        if rng.random() < self.explore_prob():
            # exploration: two distinct interior marks to map out the landscape
            a = rng.choice(pool)
            far = [c for c in pool if self._zdist(c, a) >= self.d_min]
            b = rng.choice(far) if far else rng.choice(pool)
        else:
            # exploitation: dueling Thompson -- argmax of two posterior draws
            w1 = self.sample_w(rng)
            w2 = self.sample_w(rng)
            a = self._argmax(pool, w1)
            b = (self._argmax(pool, w2, avoid=a, min_dist=self.d_min)
                 or self._argmax(pool, w2, avoid=a, min_dist=1e-9) or a)
        self._recent.append(a)
        self._recent.append(b)
        return a, b


# ---------------------------------------------------------------------------
# Demo: recover a synthetic peaked preference
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = random.Random(0)
    K = 4
    stds = [1.0] * K
    true_peak = [0.8, -0.5, 0.0, 1.2]        # preferred z per axis

    def true_util(c):
        return -sum((c[k] - true_peak[k]) ** 2 for k in range(K))

    m = PreferenceModel(stds, rng=rng)
    for _ in range(400):
        a, b = m.next_duel(rng)
        pa = _sigmoid(true_util(a) - true_util(b))
        winner = "a" if rng.random() < pa else "b"
        m.observe(a, b, winner)

    got = [round(z, 2) for z, _ in m.preferred_z()]
    print(f"true peak z*: {true_peak}")
    print(f"learned z*  : {got}  (from {m.n_obs} duels)")
