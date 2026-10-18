import unittest

from quassar_risk import (
    covariance,
    gaussian_var_es,
    historical_var_es,
    portfolio_variance,
    simulate_gaussian,
)


class TestRisk(unittest.TestCase):
    def test_covariance(self):
        self.assertEqual(covariance([[1, 2], [2, 4], [3, 6]]), [[1, 2], [2, 4]])

    def test_covariance_rejects(self):
        for rows in ([], [[1]], [[1], [1, 2]], [[1], [float("nan")]]):
            with self.subTest(rows=rows), self.assertRaises((ValueError, TypeError)):
                covariance(rows)

    def test_variance(self):
        self.assertAlmostEqual(portfolio_variance([0.5, 0.5], [[4, 1], [1, 9]]), 3.75)

    def test_negative_variance(self):
        with self.assertRaises(ValueError):
            portfolio_variance([1], [[-1]])

    def test_historical(self):
        self.assertEqual(historical_var_es([1, 2, 3, 4], 0.5), (2, 3.5))
        self.assertEqual(historical_var_es([1, 2, 3, 4], 0.75), (3, 4))
        self.assertEqual(historical_var_es([1, 2, 3, 4], 0.875), (4, 4))

    def test_historical_rejects(self):
        for a in (0, 1, -0.1, float("nan")):
            with self.assertRaises(ValueError):
                historical_var_es([1], a)

    def test_gaussian(self):
        v, e = gaussian_var_es(0, 1, 0.975)
        self.assertAlmostEqual(v, 1.959963984540054, places=10)
        self.assertAlmostEqual(e, 2.337802792201413, places=10)

    def test_gaussian_degenerate(self):
        self.assertEqual(gaussian_var_es(3, 0), (3, 3))

    def test_simulation_reproducible(self):
        a = simulate_gaussian([0, 0], [[1, 0.3], [0.3, 1]], 100, 42)
        self.assertEqual(a, simulate_gaussian([0, 0], [[1, 0.3], [0.3, 1]], 100, 42))
        self.assertNotEqual(a, simulate_gaussian([0, 0], [[1, 0.3], [0.3, 1]], 100, 43))

    def test_semidefinite(self):
        a = simulate_gaussian([0, 0], [[1, 1], [1, 1]], 10, 2)
        self.assertTrue(all(abs(x - y) < 1e-12 for x, y in a))

    def test_indefinite(self):
        with self.assertRaises(ValueError):
            simulate_gaussian([0, 0], [[1, 2], [2, 1]], 10, 2)

    def test_simulation_invalid(self):
        with self.assertRaises(ValueError):
            simulate_gaussian([0], [[1]], 0, 2)


if __name__ == "__main__":
    unittest.main()
