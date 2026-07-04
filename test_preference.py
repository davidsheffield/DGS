"""Tests for preference_model.py: feature map, IRLS stability, hybrid scheduler,
truncation to active axes, and peak recovery.

Run:  python3 -m unittest test_preference -v

All synthetic -- no ``Samples/`` needed.
"""

import math
import random
import unittest
from collections import Counter

from preference_model import PreferenceModel, _sigmoid, _softplus


class TestFeatureMap(unittest.TestCase):
    def test_phi_shape_and_standardization(self):
        m = PreferenceModel([2.0, 4.0])
        phi = m.phi([2.0, 8.0])            # z = [1, 2]
        self.assertEqual(len(phi), 4)      # 2K
        self.assertEqual(phi, [1.0, 2.0, 1.0, 4.0])   # [z, z^2]

    def test_zero_std_axis_is_safe(self):
        m = PreferenceModel([0.0, 1.0])    # first axis near-constant
        self.assertTrue(all(math.isfinite(x) for x in m.phi([0.0, 3.0])))


class TestSoftplus(unittest.TestCase):
    def test_matches_log1p_exp(self):
        for x in (-40.0, -1.0, 0.0, 1.0, 40.0):
            self.assertAlmostEqual(_softplus(x), math.log1p(math.exp(x)), places=9)


class TestPosteriorStability(unittest.TestCase):
    def test_bounded_weights_on_separable_data(self):
        # A always wins by a wide margin -> MLE diverges, but the Gaussian prior
        # + line search must keep the MAP finite (regression guard for the
        # undamped-Newton blow-up).
        m = PreferenceModel([1.0, 1.0], prior_var=1.0)
        for _ in range(80):
            m.observe([2.0, 0.0], [-2.0, 0.0], "a")
        self.assertTrue(all(abs(w) < 50.0 for w in m.w),
                        f"weights not bounded: {m.w}")
        self.assertTrue(all(math.isfinite(w) for w in m.w))

    def test_ties_pull_toward_indifference(self):
        m = PreferenceModel([1.0], prior_var=1.0)
        for _ in range(40):
            m.observe([1.5], [-1.5], "tie")     # y = 0.5 everywhere
        # a symmetric tie signal shouldn't produce a strong linear preference
        self.assertLess(abs(m.w[0]), 0.5)


class TestPeakRecovery(unittest.TestCase):
    def test_recovers_synthetic_peak(self):
        rng = random.Random(1)
        K = 3
        peak = [0.9, -0.6, 0.3]

        def util(c):
            return -sum((c[k] - peak[k]) ** 2 for k in range(K))

        m = PreferenceModel([1.0] * K, rng=rng)
        for _ in range(450):
            a, b, meta = m.next_duel(rng)
            p = _sigmoid(util(a) - util(b))
            m.observe(a, b, "a" if rng.random() < p else "b")

        learned = [z for z, _ in m.preferred_z()]
        for k in range(K):
            self.assertLess(abs(learned[k] - peak[k]), 0.4,
                            f"axis {k}: learned {learned[k]:.2f} vs {peak[k]}")
            self.assertTrue(m.preferred_z()[k][1], f"axis {k} should read as a peak")

    def test_recovers_synthetic_peak_hybrid_scheduler_with_active_axes(self):
        """Same recovery test, but through the full hybrid scheduler (axis +
        blend + occasional confirm duels) with n_active < K and real seed_zs,
        matching how preference_server.py actually drives the model."""
        rng = random.Random(7)
        K = 5
        M = 3
        peak = [0.7, -0.4, 0.5]           # active-axis peak only
        seed_zs = [[rng.gauss(0.0, 1.0) for _ in range(K)] for _ in range(6)]

        def util(c):
            z = [c[k] / 1.0 for k in range(M)]
            return -sum((z[k] - peak[k]) ** 2 for k in range(M))

        m = PreferenceModel([1.0] * K, seed_zs=seed_zs, n_active=M, rng=rng)
        for i in range(500):
            mode = "confirm" if i % 25 == 24 else None
            a, b, meta = m.next_duel(rng, mode=mode)
            p = _sigmoid(util(a) - util(b))
            m.observe(a, b, "a" if rng.random() < p else "b")

        learned = [z for z, _ in m.preferred_z()]
        for k in range(M):
            self.assertLess(abs(learned[k] - peak[k]), 0.45,
                            f"axis {k}: learned {learned[k]:.2f} vs {peak[k]}")


class TestPreferredZ(unittest.TestCase):
    def test_interior_peak_formula(self):
        m = PreferenceModel([1.0])
        m.w = [1.0, -0.5]                  # lin=1, quad=-0.5 -> z* = -1/(2*-0.5)=1
        (z, peak), = m.preferred_z()
        self.assertTrue(peak)
        self.assertAlmostEqual(z, 1.0, places=6)

    def test_edge_preference_flagged(self):
        m = PreferenceModel([1.0], z_max=2.5)
        m.w = [1.0, 0.2]                   # convex -> best at the +bound
        (z, peak), = m.preferred_z()
        self.assertFalse(peak)
        self.assertAlmostEqual(z, 2.5, places=6)


class TestBestCoeffs(unittest.TestCase):
    def test_non_peaked_axis_zero_vs_edge(self):
        m = PreferenceModel([2.0, 4.0], z_max=2.5)
        # axis 0: concave (lin=1, quad=-0.5) -> interior peak at z*=1
        # axis 1: convex (lin=1, quad=0.2) -> no interior peak, edge at +z_max
        m.w = [1.0, 1.0, -0.5, 0.2]
        best = m.best_coeffs()
        pref = m.preferred_coeffs()

        # Peaked axis: best_coeffs and preferred_coeffs agree.
        self.assertAlmostEqual(best[0], pref[0], places=6)
        self.assertAlmostEqual(best[0], 1.0 * 2.0, places=6)   # z* * std

        # Non-peaked axis: best_coeffs sits at the mean (0); preferred_coeffs
        # sits at the +z_max edge -- that edge is a prior artifact, not a
        # learned preference, which is exactly the difference best_coeffs fixes.
        self.assertAlmostEqual(best[1], 0.0, places=6)
        self.assertAlmostEqual(pref[1], 2.5 * 4.0, places=6)
        self.assertNotAlmostEqual(best[1], pref[1], places=3)


class TestAxisSelectionWeighting(unittest.TestCase):
    def test_high_variance_axes_chosen_more_and_selection_not_confined_to_top3(self):
        # Strongly descending stds: axis 0 should dominate the staircase
        # schedule, axis 5 (lowest variance) should be picked least, and with
        # 6 axes some duels must land beyond the old hard-coded top-3.
        stds = [5.0, 3.0, 1.5, 0.7, 0.3, 0.1]
        rng = random.Random(42)
        m = PreferenceModel(stds, n_active=None, rng=rng)

        # Warm the model with some blend-duel observations so zstar_stds()
        # has something other than a totally flat prior to work with.
        for _ in range(15):
            a, b, meta = m.next_duel(rng, mode="blend")
            m.observe(a, b, rng.choice(["a", "b"]))

        counts = Counter()
        for _ in range(200):
            a, b, meta = m.next_duel(rng, mode="axis")
            counts[meta["axis"]] += 1

        self.assertGreater(counts[0], counts[5],
                           f"highest-std axis should be probed more than the "
                           f"lowest-std axis: {dict(counts)}")
        self.assertTrue(any(k >= 3 for k in counts),
                        f"selection should not be confined to a top-3: {dict(counts)}")


class TestNextDuel(unittest.TestCase):
    def test_distinct_and_in_bounds(self):
        rng = random.Random(2)
        stds = [1.0, 2.0, 0.5]
        m = PreferenceModel(stds, z_max=2.5, rng=rng)
        for _ in range(20):
            a, b, meta = m.next_duel(rng)
            self.assertEqual(len(a), 3)
            self.assertNotEqual(a, b)
            for c in (a, b):
                for ck, s in zip(c, stds):
                    self.assertLessEqual(abs(ck), 2.5 * s + 1e-9)


class TestActiveAxisTruncation(unittest.TestCase):
    def test_tail_beyond_n_active_does_not_affect_fit(self):
        # K=5, only the first 2 axes active; changing tail entries of an
        # observed vote must not change what gets fit.
        stds = [1.0] * 5
        m1 = PreferenceModel(stds, n_active=2, rng=random.Random(0))
        m2 = PreferenceModel(stds, n_active=2, rng=random.Random(0))
        a1 = [0.5, -0.3, 0.0, 0.0, 0.0]
        b1 = [-0.5, 0.3, 0.0, 0.0, 0.0]
        a2 = [0.5, -0.3, 9.0, -7.0, 3.0]     # same active axes, wild tail
        b2 = [-0.5, 0.3, -4.0, 8.0, -1.0]
        m1.observe(a1, b1, "a")
        m2.observe(a2, b2, "a")
        self.assertTrue(all(math.isfinite(w) for w in m1.w))
        self.assertTrue(all(math.isfinite(w) for w in m2.w))
        self.assertEqual(m1.w, m2.w)

    def test_observe_with_full_k_vectors_stays_finite(self):
        stds = [1.0, 2.0, 0.5, 3.0]
        m = PreferenceModel(stds, n_active=2, rng=random.Random(3))
        rng = random.Random(4)
        for _ in range(50):
            a = [rng.gauss(0, 1) for _ in range(4)]
            b = [rng.gauss(0, 1) for _ in range(4)]
            m.observe(a, b, rng.choice(["a", "b", "tie"]))
        self.assertTrue(all(math.isfinite(w) for w in m.w))


class TestHybridScheduler(unittest.TestCase):
    def _make(self, n_active=3, seed_zs=None, seed=0):
        stds = [3.0, 2.0, 1.0, 0.5, 0.25]
        return PreferenceModel(stds, n_active=n_active, seed_zs=seed_zs,
                               rng=random.Random(seed)), stds

    def test_axis_duels_differ_on_exactly_one_active_axis_and_zero_tail(self):
        m, stds = self._make(n_active=3)
        # Warm up so preferred_z / zstar_stds have something to work with.
        rng = random.Random(11)
        for _ in range(15):
            a, b, meta = m.next_duel(rng, mode="blend")
            m.observe(a, b, rng.choice(["a", "b"]))

        for _ in range(10):
            a, b, meta = m.next_duel(rng, mode="axis")
            self.assertEqual(meta["mode"], "axis")
            self.assertIn("axis", meta)
            k = meta["axis"]
            self.assertLess(k, m.M)
            # tails beyond n_active are zero
            for c in (a, b):
                for j in range(m.M, m.K):
                    self.assertEqual(c[j], 0.0)
            # differ on exactly axis k among the active axes
            diffs = [j for j in range(m.M) if abs(a[j] - b[j]) > 1e-9]
            self.assertEqual(diffs, [k])

    def test_blend_duels_zero_tail_and_finite(self):
        m, stds = self._make(n_active=2)
        rng = random.Random(5)
        for _ in range(5):
            a, b, meta = m.next_duel(rng, mode="blend")
            self.assertEqual(meta["mode"], "blend")
            for c in (a, b):
                self.assertTrue(all(math.isfinite(x) for x in c))
                for j in range(m.M, m.K):
                    self.assertEqual(c[j], 0.0)
                for j in range(m.M):
                    self.assertLessEqual(abs(c[j]), m.z_max * stds[j] + 1e-9)

    def test_blend_duels_with_seed_zs_stay_sane(self):
        rng = random.Random(6)
        seed_zs = [[rng.gauss(0, 1) for _ in range(5)] for _ in range(6)]
        m, stds = self._make(n_active=3, seed_zs=seed_zs, seed=6)
        for _ in range(20):
            a, b, meta = m.next_duel(rng, mode="blend")
            for c in (a, b):
                self.assertTrue(all(math.isfinite(x) for x in c))
                for j in range(m.M, m.K):
                    self.assertEqual(c[j], 0.0)
                for j in range(m.M):
                    self.assertLessEqual(abs(c[j]), m.z_max * stds[j] + 1e-9)

    def test_confirm_mode_returns_base_point_as_one_side(self):
        m, stds = self._make(n_active=3)
        rng = random.Random(21)
        # Feed some votes so preferred_z / base point aren't trivially all-zero.
        for _ in range(30):
            a, b, meta = m.next_duel(rng, mode="blend")
            m.observe(a, b, rng.choice(["a", "b"]))

        base = m._coeffs_from_z(m._base_z())
        for _ in range(10):
            a, b, meta = m.next_duel(rng, mode="confirm")
            self.assertEqual(meta["mode"], "confirm")
            self.assertTrue(a == base or b == base,
                            f"neither side equals the base point: a={a} b={b} base={base}")


if __name__ == "__main__":
    unittest.main()
