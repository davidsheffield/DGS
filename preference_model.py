"""Bayesian *peaked* preference model over the eigenshape space.

The evolver (``evolve_server.py``) asks you to pick which grid cells to breed;
this instead learns, from forced A/B duels, **what values along each eigen-axis
you prefer** -- and, crucially, that the preference has a *peak*: being on either
side of a sweet spot is worse, so votes can drift back and forth near the
optimum without a clear winner.

Model
-----
A candidate is a coefficient vector ``c`` in an ``eigen.PCABasis`` (length K =
``basis.n_components``).  By default **all K axes are learned and actively
varied** -- ``n_active`` = M = K.  The scheduler doesn't spend its budget
evenly, though: it's variance-weighted, so the leading (highest-``stds``) axes
get most of the staircase duels while low-variance tail axes still get
occasional probes (see "axis" below).  ``n_active`` remains available to
*truncate* the model to the leading M < K axes when that's wanted (tests, the
demo below) -- tail axes beyond M then sit at 0 (the mean) in every generated
candidate, and logged votes still carry the full K-length coefficient vector
(so nothing is lost if M is later raised) -- ``phi``/``observe``/``utility``
silently truncate to the first M entries.  Standardize by the per-component
population std so a unit is comparable across axes::

    z_k = c_k / stds_k          (k = 0 .. M-1)

and use a **per-axis quadratic** feature map::

    phi(c) = [z_1..z_M, z_1^2 .. z_M^2]            (length 2M)

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
2M is small, so a pure-Python Cholesky solve is ample.

Active sampling: a hybrid scheduler
------------------------------------
Pure dueling-Thompson over the whole box turned out to spend most of its
budget confirming things already known.  ``next_duel`` instead mixes three
kinds of question, each answering something the others can't:

* **"axis" (staircase)** -- hold every axis but one at its current best guess
  and duel two points that straddle the current guess on one axis, chosen by
  **weighted random sampling** over all M active axes: weight = posterior
  uncertainty of that axis's z* (``zstar_stds``) times how much a z-unit on
  that axis moves the shape (``stds``).  This is what makes high-variance,
  still-uncertain axes get most of the staircase budget while settled axes
  (small ``zstar_std``) and low-variance tail axes fade out -- without ever
  being excluded outright, so a tail axis still gets the occasional probe.
  This isolates one axis at a time, the fastest way to pin down its curvature.
* **"blend"** -- duel two broad, whole-shape candidates built by Dirichlet-
  blending real seed marks (plus jitter) over the active axes, so votes also
  cover combinations the axis-by-axis view can't reach.  Chosen by the argmax
  of two independent posterior draws (dueling Thompson), same idea as before.
* **"confirm"** -- duel the current believed-best mark against a near
  neighbour (the mean, a seed, or a jittered variant), to sanity-check
  convergence.  Available on request (``mode="confirm"``), not part of the
  random schedule.

``next_duel(rng, mode=None)`` picks "blend" vs. "axis" with an annealed
probability (``blend_prob``, née ``explore_prob``): early on, with a diffuse
posterior, broad blend duels teach the model where the interior lives at all;
once curvature is known, axis duels dominate to sharpen each optimum.  A
recency buffer (used by "blend") forbids showing anything close to the last
few marks so a shape only returns after a palette cleanser; "axis"/"confirm"
duels intentionally reuse the current base point and bypass it.

Pure standard library, like ``genome.py`` / ``eigen.py``.
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

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
    def __init__(self, stds: Sequence[float], *,
                 seed_zs: Optional[Sequence[Sequence[float]]] = None,
                 n_active: Optional[int] = None,
                 prior_var: float = 1.0,
                 z_max: float = 2.5, pool_size: int = 200,
                 recent: int = 12, d_min: float = 1.5,
                 rng: random.Random | None = None):
        # Guard near-constant axes so z = c/std stays finite.
        self.stds = [s if abs(s) > 1e-9 else 1e-9 for s in stds]
        self.K = len(self.stds)
        # n_active (M): leading axes the model learns/varies; stds is already
        # variance-sorted descending, so "leading" == "most important".
        self.M = self.K if n_active is None else max(1, min(n_active, self.K))
        self.D = 2 * self.M
        # Standardized (z-unit) seed coefficient vectors, full K-length, used
        # by "blend" candidates. Kept as given; truncated to M on use.
        self.seed_zs: List[List[float]] = [list(z) for z in seed_zs] if seed_zs else []
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
        """Standardize and truncate to the M active axes. ``coeffs`` may be a
        full K-length vector (as logged) -- the ``zip`` stops at M, so tail
        entries are simply dropped, not fit against."""
        return [c / s for c, s in zip(coeffs, self.stds[:self.M])]

    def phi(self, coeffs: Sequence[float]) -> List[float]:
        z = self._z(coeffs)
        return z + [zi * zi for zi in z]

    def utility(self, coeffs: Sequence[float],
                w: Sequence[float] | None = None) -> float:
        w = self.w if w is None else w
        ph = self.phi(coeffs)
        return sum(w[j] * ph[j] for j in range(self.D))

    # --- z (active, length M) <-> full K-length coeffs -----------------------
    def _coeffs_from_z(self, z_active: Sequence[float]) -> List[float]:
        """Active-axis z-values -> full K-length coefficient vector, tail
        zero-padded (0 == the mean on axes the model doesn't touch)."""
        out = [0.0] * self.K
        for k, z in enumerate(z_active):
            out[k] = z * self.stds[k]
        return out

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
    def _zstar_for(self, w: Sequence[float], k: int) -> Tuple[float, bool]:
        """(z*, is_interior_peak) for active axis k under weight vector w."""
        b = w[k]                         # linear weight
        a = w[self.M + k]                # quadratic weight
        if a < -1e-9:                     # concave -> interior peak
            z = max(-self.z_max, min(self.z_max, -b / (2.0 * a)))
            return z, True
        # convex/linear -> best endpoint
        zp, zm = self.z_max, -self.z_max
        z = zp if (b * zp + a * zp * zp) >= (b * zm + a * zm * zm) else zm
        return z, False

    def preferred_z(self) -> List[Tuple[float, bool]]:
        """Per active axis: (preferred standardized value z*, is_interior_peak)."""
        return [self._zstar_for(self.w, k) for k in range(self.M)]

    def preferred_coeffs(self) -> List[float]:
        """Full K-length coefficient vector at the model's optimum (0 on axes
        beyond n_active)."""
        z = [zk for zk, _ in self.preferred_z()]
        return self._coeffs_from_z(z)

    def best_coeffs(self) -> List[float]:
        """Full K-length coefficient vector at the model's *current best guess*:
        peaked axes at their learned z*, axes with no interior peak yet at 0.0
        (the mean), rather than the +-z_max edge ``preferred_coeffs`` reports
        for those axes. ``preferred_coeffs`` is meant to describe "the model's
        optimum" for picking a duel candidate, and pushing a not-yet-curved
        axis to the sampling boundary is the right move *there* -- it's
        exploration, keeping that axis in play until votes reveal a peak. For
        a "what does the mark look like right now" display, that same edge
        value would misrepresent an artifact of the flat prior as a learned
        preference; ``best_coeffs`` leaves such axes at the mean instead, so it
        only shows what the votes have actually pinned down so far."""
        return self._coeffs_from_z(self._base_z())

    def _base_z(self) -> List[float]:
        """Active-axis z of the model's current best guess: peaked axes sit at
        their z*; axes with no interior peak yet sit at 0 (the mean) rather
        than the +-z_max edge ``preferred_z`` picks for a not-yet-curved axis
        (that edge is an artifact of a flat prior, not a real preference)."""
        return [z if is_peak else 0.0 for z, is_peak in self.preferred_z()]

    def zstar_stds(self, n_draws: int = 32) -> List[float]:
        """Posterior uncertainty of each active axis's z*: draw ``n_draws``
        weight vectors from the Laplace posterior, compute z* per draw, take
        the std. Used both for the axis report and to pick which axis the
        "axis" duel should probe next."""
        draws = [self.sample_w(self.rng) for _ in range(n_draws)]
        out = []
        for k in range(self.M):
            zs = [self._zstar_for(w, k)[0] for w in draws]
            mean = sum(zs) / len(zs)
            var = sum((z - mean) ** 2 for z in zs) / len(zs)
            out.append(math.sqrt(var))
        return out

    def axis_report(self) -> List[dict]:
        """One dict per *active* axis for the analysis view, sorted by how
        strongly the votes constrain it (utility span across the axis)."""
        pref = self.preferred_z()
        zstd = self.zstar_stds()
        rows = []
        for k in range(self.M):
            b = self.w[k]
            a = self.w[self.M + k]
            us = [b * z + a * z * z
                  for z in (i / 20.0 * self.z_max for i in range(-20, 21))]
            rows.append({
                "axis": k,
                "z_star": pref[k][0],
                "peak": pref[k][1],
                "lin": b, "quad": a,
                "lin_std": self.weight_std(k),
                "quad_std": self.weight_std(self.M + k),
                "zstar_std": zstd[k],
                "span": max(us) - min(us),
            })
        rows.sort(key=lambda r: r["span"], reverse=True)
        return rows

    # --- shared duel-picking machinery ---------------------------------------
    def _zdist(self, c1: Sequence[float], c2: Sequence[float]) -> float:
        return math.sqrt(sum(((x - y) / s) ** 2
                             for x, y, s in zip(c1, c2, self.stds)))

    def _too_recent(self, coeffs: Sequence[float]) -> bool:
        return any(self._zdist(coeffs, r) < self.d_min for r in self._recent)

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

    def _dirichlet3(self, rng: random.Random) -> List[float]:
        """Symmetric Dirichlet(1,1,1) via normalized Exp(1) draws (stdlib-only:
        that's the standard construction -- a Gamma(1,*) is an Exp, and
        normalizing i.i.d. Gammas of a shared scale gives a Dirichlet)."""
        e = [rng.expovariate(1.0) for _ in range(3)]
        s = sum(e)
        return [ei / s for ei in e]

    # --- "blend" candidates ---------------------------------------------------
    def _blend_candidate(self, rng: random.Random) -> List[float]:
        """One whole-shape candidate: a Dirichlet-weighted blend of 3 distinct
        seed marks (in z units, active axes only) plus jitter; falls back to
        the plain N(0,1) interior sampling this replaced when there aren't
        enough seeds to blend."""
        if len(self.seed_zs) >= 3:
            idxs = rng.sample(range(len(self.seed_zs)), 3)
            w = self._dirichlet3(rng)
            z = [sum(w[i] * self.seed_zs[idxs[i]][k] for i in range(3))
                 for k in range(self.M)]
            z = [zk + rng.gauss(0.0, 0.3) for zk in z]
        else:
            z = [rng.gauss(0.0, 1.0) for _ in range(self.M)]
        z = [max(-self.z_max, min(self.z_max, zk)) for zk in z]
        return self._coeffs_from_z(z)

    def _blend_pool(self, rng: random.Random) -> List[List[float]]:
        pool = [c for c in (self._blend_candidate(rng)
                            for _ in range(self.pool_size)) if not self._too_recent(c)]
        if not pool:                      # recency starved the pool: relax it
            pool = [self._blend_candidate(rng) for _ in range(self.pool_size)]
        return pool

    def _blend_duel(self, rng: random.Random) -> Tuple[List[float], List[float], dict]:
        pool = self._blend_pool(rng)
        w1 = self.sample_w(rng)
        w2 = self.sample_w(rng)
        a = self._argmax(pool, w1)
        b = (self._argmax(pool, w2, avoid=a, min_dist=self.d_min)
             or self._argmax(pool, w2, avoid=a, min_dist=1e-9) or a)
        self._recent.append(a)
        self._recent.append(b)
        return a, b, {"mode": "blend"}

    # --- "axis" (staircase) duels ---------------------------------------------
    def _axis_duel(self, rng: random.Random) -> Tuple[List[float], List[float], dict]:
        """Hold every active axis at the current base point except one, and
        straddle that axis's current best guess. The axis is picked by
        **weighted random sampling** over all M active axes, weight =
        ``zstar_std[k] * stds[k]`` (posterior uncertainty of that axis's z*,
        times how much a z-unit on that axis matters geometrically) -- so
        high-variance, still-uncertain axes get most of the staircase budget,
        settled axes fade out, and low-variance tail axes still get occasional
        probes rather than being excluded outright."""
        zstd = self.zstar_stds()
        scores = [zstd[k] * self.stds[k] for k in range(self.M)]
        weights = [s if math.isfinite(s) and s > 0.0 else 0.0 for s in scores]
        if any(weights):
            k = rng.choices(range(self.M), weights=weights)[0]
        else:
            # All scores are ~0 (or non-finite) -- e.g. a totally fresh model
            # with no votes yet -- fall back to a uniform pick.
            k = rng.choice(range(self.M))

        base_z = self._base_z()
        z_star, is_peak = self.preferred_z()[k]
        if is_peak:
            t = z_star
            delta = max(0.3, min(1.2, zstd[k]))
        else:
            # no interior peak learned yet: probe the interior to learn curvature
            t = 0.0
            delta = 1.25

        lo = max(-self.z_max, t - delta)
        hi = min(self.z_max, t + delta)
        if hi - lo < 1e-6:
            # clamping collapsed the pair onto one point: recenter around 0
            # so the two candidates stay distinct.
            lo = max(-self.z_max, -delta)
            hi = min(self.z_max, delta)

        z_a, z_b = (lo, hi) if rng.random() < 0.5 else (hi, lo)
        za = list(base_z); za[k] = z_a
        zb = list(base_z); zb[k] = z_b
        return self._coeffs_from_z(za), self._coeffs_from_z(zb), {"mode": "axis", "axis": k}

    # --- "confirm" duels --------------------------------------------------------
    def _confirm_duel(self, rng: random.Random) -> Tuple[List[float], List[float], dict]:
        """Duel the current believed-best mark against a near neighbour, to
        sanity-check convergence rather than explore further."""
        base_z = self._base_z()
        pick = rng.choice(("mean", "seed", "blend"))
        if pick == "seed" and self.seed_zs:
            sz = rng.choice(self.seed_zs)
            other_z = [sz[k] if k < len(sz) else 0.0 for k in range(self.M)]
        elif pick == "blend":
            # a blend-pool candidate near the optimum: jitter centered on the
            # base point itself, rather than on a random blend.
            other_z = [base_z[k] + rng.gauss(0.0, 0.4) for k in range(self.M)]
        else:
            other_z = [0.0] * self.M
        other_z = [max(-self.z_max, min(self.z_max, z)) for z in other_z]

        a_coeffs = self._coeffs_from_z(base_z)
        b_coeffs = self._coeffs_from_z(other_z)
        if rng.random() < 0.5:
            a_coeffs, b_coeffs = b_coeffs, a_coeffs
        return a_coeffs, b_coeffs, {"mode": "confirm"}

    # --- scheduler ------------------------------------------------------------
    def blend_prob(self) -> float:
        """Fraction of scheduled duels that should be "blend" (broad,
        seed-blend, whole-shape) rather than "axis" (staircase) duels,
        annealed with data.

        (Formerly ``explore_prob``: same annealing, repurposed for the
        axis-vs-blend mode choice instead of exploit-vs-explore within a
        single dueling-Thompson pool.) While the posterior is flat, blend
        duels are needed to find out *where* the interior peaks even are;
        once curvature is known on the active axes, axis duels spend the
        budget sharpening each one instead.
        """
        return max(0.25, math.exp(-self.n_obs / 40.0))

    def next_duel(self, rng: random.Random | None = None,
                  mode: Optional[str] = None
                  ) -> Tuple[List[float], List[float], dict]:
        """Pick the next duel. Returns (a_coeffs, b_coeffs, meta), all
        full K-length coefficient vectors (tail zero beyond n_active).

        ``mode``: None picks "blend" vs. "axis" via ``blend_prob()``; pass
        "axis", "blend", or "confirm" to force that kind of duel.
        """
        rng = rng or self.rng
        if mode is None:
            mode = "blend" if rng.random() < self.blend_prob() else "axis"
        if mode == "axis":
            return self._axis_duel(rng)
        if mode == "blend":
            return self._blend_duel(rng)
        if mode == "confirm":
            return self._confirm_duel(rng)
        raise ValueError(f'mode must be "axis", "blend", "confirm" or None, got {mode!r}')


# ---------------------------------------------------------------------------
# Demo: recover a synthetic peaked preference, via the hybrid scheduler
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = random.Random(0)
    K = 6
    M = 4                                     # only the leading 4 axes are active
    stds = [1.0] * K
    true_peak = [0.8, -0.5, 0.0, 1.2]         # preferred z, active axes only

    # A handful of synthetic "seeds" in z-units, for the blend duels to mix.
    seed_zs = [[rng.gauss(0.0, 1.0) for _ in range(K)] for _ in range(5)]

    def true_util(c):
        z = [c[k] / stds[k] for k in range(M)]
        return -sum((z[k] - true_peak[k]) ** 2 for k in range(M))

    m = PreferenceModel(stds, seed_zs=seed_zs, n_active=M, rng=rng)
    for _ in range(400):
        a, b, meta = m.next_duel(rng)
        pa = _sigmoid(true_util(a) - true_util(b))
        winner = "a" if rng.random() < pa else "b"
        m.observe(a, b, winner)

    got = [round(z, 2) for z, _ in m.preferred_z()]
    print(f"true peak z*: {true_peak}")
    print(f"learned z*  : {got}  (from {m.n_obs} duels, {m.M} active of {m.K} axes)")
