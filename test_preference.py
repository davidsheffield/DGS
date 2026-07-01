"""Tests for preference_model.py: feature map, IRLS stability, peak recovery.

Run:  python3 -m unittest test_preference -v

All synthetic -- no ``Samples/`` needed.
"""

import math
import random
import unittest

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
            a, b = m.next_duel(rng)
            p = _sigmoid(util(a) - util(b))
            m.observe(a, b, "a" if rng.random() < p else "b")

        learned = [z for z, _ in m.preferred_z()]
        for k in range(K):
            self.assertLess(abs(learned[k] - peak[k]), 0.4,
                            f"axis {k}: learned {learned[k]:.2f} vs {peak[k]}")
            self.assertTrue(m.preferred_z()[k][1], f"axis {k} should read as a peak")


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


class TestNextDuel(unittest.TestCase):
    def test_distinct_and_in_bounds(self):
        rng = random.Random(2)
        stds = [1.0, 2.0, 0.5]
        m = PreferenceModel(stds, z_max=2.5, rng=rng)
        for _ in range(20):
            a, b = m.next_duel(rng)
            self.assertEqual(len(a), 3)
            self.assertNotEqual(a, b)
            for c in (a, b):
                for ck, s in zip(c, stds):
                    self.assertLessEqual(abs(ck), 2.5 * s + 1e-9)


if __name__ == "__main__":
    unittest.main()
