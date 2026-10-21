"""Independent analytical and validation tests for the EVT module."""

import math
import unittest

from quassar_risk.extreme_value import GPDModel, fit_exponential_tail


class TestGPDModel(unittest.TestCase):
    def test_survival_at_threshold(self):
        model = GPDModel(10.0, 2.0, 0.2, 0.1)
        self.assertAlmostEqual(model.survival(10.0), 0.1)

    def test_exponential_survival(self):
        model = GPDModel(10.0, 2.0, 0.0, 0.1)
        self.assertAlmostEqual(
            model.survival(12.0),
            0.1 * math.exp(-1.0),
        )

    def test_gpd_survival(self):
        model = GPDModel(10.0, 2.0, 0.25, 0.1)
        expected = 0.1 * (1.0 + 0.25 * 4.0 / 2.0) ** (-4.0)
        self.assertAlmostEqual(model.survival(14.0), expected)

    def test_var_exponential_analytical(self):
        model = GPDModel(10.0, 2.0, 0.0, 0.1)
        self.assertAlmostEqual(
            model.value_at_risk(0.99),
            10.0 + 2.0 * math.log(10.0),
        )

    def test_var_survival_identity(self):
        for shape in (0.0, 0.1, 0.4, 0.8):
            with self.subTest(shape=shape):
                model = GPDModel(10.0, 2.0, shape, 0.1)
                for confidence in (0.95, 0.99, 0.999):
                    var = model.value_at_risk(confidence)
                    self.assertAlmostEqual(
                        model.survival(var),
                        1.0 - confidence,
                        delta=1e-12,
                    )

    def test_expected_shortfall_exponential(self):
        model = GPDModel(10.0, 2.0, 0.0, 0.1)
        var = model.value_at_risk(0.99)
        self.assertAlmostEqual(
            model.expected_shortfall(0.99),
            var + 2.0,
        )

    def test_expected_shortfall_gpd(self):
        model = GPDModel(10.0, 2.0, 0.25, 0.1)
        var = model.value_at_risk(0.99)
        expected = var + (2.0 + 0.25 * (var - 10.0)) / 0.75
        self.assertAlmostEqual(
            model.expected_shortfall(0.99),
            expected,
        )

    def test_expected_shortfall_exceeds_var(self):
        model = GPDModel(0.0, 1.0, 0.5, 0.2)
        for confidence in (0.9, 0.99, 0.999):
            self.assertGreater(
                model.expected_shortfall(confidence),
                model.value_at_risk(confidence),
            )

    def test_var_increases_with_confidence(self):
        model = GPDModel(0.0, 1.0, 0.2, 0.1)
        levels = (0.91, 0.95, 0.99, 0.999)
        values = [model.value_at_risk(p) for p in levels]
        self.assertEqual(values, sorted(values))
        self.assertEqual(len(set(values)), len(values))

    def test_invalid_scale(self):
        with self.assertRaises(ValueError):
            GPDModel(0.0, 0.0, 0.2, 0.1)

    def test_invalid_shape(self):
        for shape in (-0.1, 1.0, float("nan")):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                GPDModel(0.0, 1.0, shape, 0.1)

    def test_invalid_exceedance_probability(self):
        for probability in (0.0, -0.1, 1.1):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                GPDModel(0.0, 1.0, 0.2, probability)

    def test_confidence_outside_tail(self):
        model = GPDModel(0.0, 1.0, 0.2, 0.1)
        for confidence in (0.5, 0.9, 1.0):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                model.value_at_risk(confidence)

    def test_invalid_survival_input(self):
        model = GPDModel(10.0, 2.0, 0.2, 0.1)
        for loss in (9.0, float("nan"), float("inf")):
            with self.subTest(loss=loss), self.assertRaises(ValueError):
                model.survival(loss)


class TestExponentialTailEstimator(unittest.TestCase):
    def test_known_sample(self):
        model = fit_exponential_tail(
            [0.0, 1.0, 2.0, 4.0, 6.0],
            threshold=2.0,
        )
        self.assertAlmostEqual(model.scale, 3.0)
        self.assertAlmostEqual(model.shape, 0.0)
        self.assertAlmostEqual(model.exceedance_probability, 0.4)

    def test_no_exceedances(self):
        with self.assertRaises(ValueError):
            fit_exponential_tail([1.0, 2.0], threshold=3.0)

    def test_empty_sample(self):
        with self.assertRaises(ValueError):
            fit_exponential_tail([], threshold=1.0)

    def test_nonfinite_observation(self):
        with self.assertRaises(ValueError):
            fit_exponential_tail([1.0, float("nan")], threshold=0.0)


if __name__ == "__main__":
    unittest.main()
