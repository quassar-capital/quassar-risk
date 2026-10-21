"""Adversarial tests for EWMA initial covariance validation."""

import unittest

from quassar_risk.dynamic_covariance import ewma_covariance


class TestEWMAInitialPSD(unittest.TestCase):
    def test_rejects_negative_variance(self):
        with self.assertRaises(ValueError):
            ewma_covariance(
                [[1.0]],
                initial_covariance=[[-1.0]],
            )

    def test_rejects_indefinite_matrix(self):
        with self.assertRaises(ValueError):
            ewma_covariance(
                [[1.0, 1.0]],
                initial_covariance=[[1.0, 2.0], [2.0, 1.0]],
            )

    def test_accepts_singular_psd_matrix(self):
        result = ewma_covariance(
            [[0.0, 0.0]],
            decay=1.0,
            initial_covariance=[[1.0, 1.0], [1.0, 1.0]],
        )
        self.assertEqual(result, [[1.0, 1.0], [1.0, 1.0]])

    def test_rejects_zero_diagonal_with_nonzero_covariance(self):
        with self.assertRaises(ValueError):
            ewma_covariance(
                [[0.0, 0.0]],
                initial_covariance=[[0.0, 1.0], [1.0, 1.0]],
            )


if __name__ == "__main__":
    unittest.main()
