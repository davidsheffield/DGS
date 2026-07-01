"""Tests for eigen.py: Jacobi solver, basis fit, round-trip, breeding.

Run:  python3 -m unittest test_eigen -v

The round-trip and breeding tests need the seed SVGs in ``Samples/`` and are
skipped if they're absent (Samples/ is gitignored data).
"""

import math
import os
import random
import unittest

import eigen
from eigen import (PCABasis, breed_coeffs, feature_layout, flatten,
                   jacobi_eigh, unflatten)
from genome import CANVAS_H, CANVAS_W, MIN_HANDLE, PATH_ORDER, SEGMENTS, load_samples

HAVE_SAMPLES = bool(__import__("glob").glob("Samples/vector_*.svg"))


def _load():
    return load_samples()


class TestLayout(unittest.TestCase):
    def test_dimension(self):
        layout = feature_layout()
        expect = sum((SEGMENTS[p] + 1) * 2 + 4 + (SEGMENTS[p] - 1) * 3
                     for p in PATH_ORDER)
        self.assertEqual(len(layout), expect)
        self.assertEqual(len(layout), 49)

    def test_len_slots_follow_angle_slots(self):
        # PCABasis.fit assumes layout[j+1]/[j+2] are the angle's len_in/len_out
        layout = feature_layout()
        for j in eigen.angle_slots(layout):
            self.assertEqual(layout[j + 1]["kind"], "len_in")
            self.assertEqual(layout[j + 2]["kind"], "len_out")
            self.assertEqual(layout[j]["index"], layout[j + 1]["index"])


class TestJacobi(unittest.TestCase):
    def test_random_symmetric(self):
        rng = random.Random(42)
        n = 8
        A = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                A[i][j] = A[j][i] = rng.uniform(-2, 2)
        evals, evecs = jacobi_eigh(A)

        # descending order
        self.assertEqual(evals, sorted(evals, reverse=True))
        # trace preserved
        self.assertAlmostEqual(sum(evals), sum(A[i][i] for i in range(n)), places=9)
        # A v = lambda v, orthonormal vectors
        for lam, v in zip(evals, evecs):
            av = [sum(A[r][c] * v[c] for c in range(n)) for r in range(n)]
            for r in range(n):
                self.assertAlmostEqual(av[r], lam * v[r], places=9)
        for i, vi in enumerate(evecs):
            for j, vj in enumerate(evecs):
                dot = sum(x * y for x, y in zip(vi, vj))
                self.assertAlmostEqual(dot, 1.0 if i == j else 0.0, places=9)


@unittest.skipUnless(HAVE_SAMPLES, "Samples/vector_*.svg not present")
class TestBasis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pop = _load()
        cls.basis = PCABasis.fit(cls.pop)

    def test_shape(self):
        self.assertEqual(self.basis.dim, 49)
        self.assertLessEqual(self.basis.n_components, len(self.pop) - 1)
        self.assertEqual(self.basis.stds, sorted(self.basis.stds, reverse=True))

    def test_components_orthonormal(self):
        for i, pi in enumerate(self.basis.components):
            for j, pj in enumerate(self.basis.components):
                dot = sum(a * b for a, b in zip(pi, pj))
                self.assertAlmostEqual(dot, 1.0 if i == j else 0.0, places=8)

    def test_round_trip_all_seeds(self):
        ref = self.basis._mean_angles()
        for g in self.pop:
            g2 = self.basis.decode(self.basis.encode(g))
            for a, b in zip(flatten(g, ref), flatten(g2, ref)):
                self.assertAlmostEqual(a, b, places=6)

    def test_round_trip_survives_svg_render(self):
        # decode -> to_svg -> re-parse stays within the 2-decimal _f formatting
        from genome import Genome
        g = self.pop[0]
        g2 = Genome.from_svg(self.basis.decode(self.basis.encode(g)).to_svg())
        for pid in PATH_ORDER:
            for (x1, y1), (x2, y2) in zip(g.paths[pid].nodes, g2.paths[pid].nodes):
                self.assertLess(abs(x1 - x2), 0.01)
                self.assertLess(abs(y1 - y2), 0.01)

    def test_dict_round_trip(self):
        basis2 = PCABasis.from_dict(self.basis.to_dict())
        c1 = self.basis.encode(self.pop[5])
        c2 = basis2.encode(self.pop[5])
        self.assertEqual(c1, c2)
        d1 = self.basis.decode(c1).to_svg()
        d2 = basis2.decode(c2).to_svg()
        self.assertEqual(d1, d2)

    def test_dict_version_check(self):
        d = self.basis.to_dict()
        d["layout_version"] = 999
        with self.assertRaises(ValueError):
            PCABasis.from_dict(d)

    def test_json_round_trip(self):
        import json
        basis2 = PCABasis.from_dict(json.loads(json.dumps(self.basis.to_dict())))
        self.assertEqual(basis2.decode(basis2.encode(self.pop[7])).to_svg(),
                         self.basis.decode(self.basis.encode(self.pop[7])).to_svg())


@unittest.skipUnless(HAVE_SAMPLES, "Samples/vector_*.svg not present")
class TestBreeding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pop = _load()
        cls.basis = PCABasis.fit(cls.pop)
        cls.coeffs = [cls.basis.encode(g) for g in cls.pop]

    def test_children_are_valid_genomes(self):
        rng = random.Random(123)
        for _ in range(50):
            a, b = rng.sample(self.coeffs, 2)
            child = self.basis.decode(breed_coeffs(a, b, self.basis.stds, rng=rng))
            for pid in PATH_ORDER:
                pg = child.paths[pid]
                self.assertEqual(len(pg.nodes), SEGMENTS[pid] + 1)
                self.assertEqual(len(pg.tangents), SEGMENTS[pid] - 1)
                for x, y in pg.nodes:
                    self.assertTrue(math.isfinite(x) and math.isfinite(y))
                    self.assertTrue(0.0 <= x <= CANVAS_W)
                    self.assertTrue(0.0 <= y <= CANVAS_H)
                for ux, uy, li, lo in pg.tangents:
                    self.assertAlmostEqual(math.hypot(ux, uy), 1.0, places=9)
                    self.assertGreaterEqual(li, MIN_HANDLE)
                    self.assertGreaterEqual(lo, MIN_HANDLE)
            # renders to parseable SVG
            child.to_svg()

    def test_no_variation_operators_returns_parent_genes(self):
        rng = random.Random(5)
        a, b = self.coeffs[0], self.coeffs[1]
        child = breed_coeffs(a, b, self.basis.stds,
                             blend_prob=0.0, rate=0.0, rng=rng)
        for cv, av, bv in zip(child, a, b):
            self.assertIn(cv, (av, bv))


if __name__ == "__main__":
    unittest.main()
