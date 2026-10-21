"""Mathematical validation of the EWMA covariance estimator."""

import math
import unittest

from quassar_risk.dynamic_covariance import ewma_covariance


class TestEWMACovariance(unittest.TestCase):
    def test_single_observation(self):
        result = ewma_covariance([[2.0, -3.0]], decay=0.5)
        expected = [[2.0, -3.0], [-3.0, 4.5]]
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(result[i][j], expected[i][j])

    def test_exact_two_step_recurrence(self):
        result = ewma_covariance([[2.0], [4.0]], decay=0.5)
        self.assertAlmostEqual(result[0][0], 9.0)

    def test_zero_decay(self):
        result = ewma_covariance([[1.0, 2.0], [3.0, 4.0]], decay=0.0)
        self.assertEqual(result, [[9.0, 12.0], [12.0, 16.0]])

    def test_unit_decay_preserves_initial(self):
        initial = [[2.0, 0.5], [0.5, 3.0]]
        result = ewma_covariance([[10.0, -10.0]], decay=1.0, initial_covariance=initial)
        self.assertEqual(result, initial)

    def test_zero_returns(self):
        result = ewma_covariance([[0.0, 0.0]], decay=0.94)
        self.assertEqual(result, [[0.0, 0.0], [0.0, 0.0]])

    def test_symmetry(self):
        result = ewma_covariance(
            [[1.0, 2.0, -1.0], [3.0, -2.0, 4.0]],
            decay=0.8,
        )
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(result[i][j], result[j][i])

    def test_positive_semidefinite_quadratic_forms(self):
        covariance = ewma_covariance(
            [[1.0, -2.0], [3.0, 1.0], [-1.0, 4.0]],
            decay=0.7,
        )
        for a in (-2.0, -1.0, 0.0, 1.0, 2.0):
            for b in (-2.0, -1.0, 0.0, 1.0, 2.0):
                quadratic = (
                    a * a * covariance[0][0]
                    + 2 * a * b * covariance[0][1]
                    + b * b * covariance[1][1]
                )
                self.assertGreaterEqual(quadratic, -1e-12)

    def test_scaling(self):
        returns = [[1.0, 2.0], [-3.0, 4.0]]
        original = ewma_covariance(returns, decay=0.8)
        scaled = ewma_covariance(
            [[5.0 * value for value in row] for row in returns],
            decay=0.8,
        )
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(scaled[i][j], 25.0 * original[i][j])

    def test_invalid_decay(self):
        for decay in (-0.1, 1.1, float("nan")):
            with self.subTest(decay=decay), self.assertRaises(ValueError):
                ewma_covariance([[1.0]], decay=decay)

    def test_empty_observations(self):
        with self.assertRaises(ValueError):
            ewma_covariance([])

    def test_ragged_observations(self):
        with self.assertRaises(ValueError):
            ewma_covariance([[1.0, 2.0], [3.0]])

    def test_nonfinite_observations(self):
        with self.assertRaises(ValueError):
            ewma_covariance([[math.inf]])

    def test_invalid_initial_dimensions(self):
        with self.assertRaises(ValueError):
            ewma_covariance(
                [[1.0, 2.0]],
                initial_covariance=[[1.0]],
            )

    def test_nonsymmetric_initial_covariance(self):
        with self.assertRaises(ValueError):
            ewma_covariance(
                [[1.0, 2.0]],
                initial_covariance=[[1.0, 2.0], [3.0, 1.0]],
            )


if __name__ == "__main__":
    unittest.main()
