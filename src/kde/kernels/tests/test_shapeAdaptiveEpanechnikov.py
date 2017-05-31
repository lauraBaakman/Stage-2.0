from unittest import TestCase

import numpy as np

from kde.kernels.shapeadaptiveepanechnikov import ShapeAdaptiveEpanechnikov, \
    _ShapeAdaptiveEpanechnikov_C, _ShapeAdaptiveEpanechnikov_Python


class TestShapeAdaptiveEpanechnikov(TestCase):
    def test_to_C_enum(self):
        actual = ShapeAdaptiveEpanechnikov.to_C_enum()
        expected = 3
        self.assertEqual(actual, expected)

    def test_default_implementation_single_pattern_l_eq_1(self):
        H = np.array([
            [0.080225998475784, 0.000182273891304],
            [0.000182273891304, 0.081385767033078]
        ])
        x = np.array([0.05, 0.05])
        local_bandwidth = 1.0
        expected = None
        actual = ShapeAdaptiveEpanechnikov(H).evaluate(x, local_bandwidth)
        self.assertAlmostEqual(actual, expected)

    def test_alternative_implementation_single_pattern_l_eq_1(self):
        H = np.array([
            [0.080225998475784, 0.000182273891304],
            [0.000182273891304, 0.081385767033078]
        ])
        x = np.array([0.05, 0.05])
        expected = None
        local_bandwidth = 1.0
        actual = ShapeAdaptiveEpanechnikov(H, implementation=_ShapeAdaptiveEpanechnikov_Python).evaluate(x, local_bandwidth)
        self.assertAlmostEqual(actual, expected)

    def test_default_implementation_single_pattern_l_neq_1(self):
        H = np.array([
            [0.080225998475784, 0.000182273891304],
            [0.000182273891304, 0.081385767033078]
        ])
        x = np.array([0.05, 0.05])
        local_bandwidth = 0.5
        expected = None
        actual = ShapeAdaptiveEpanechnikov(H).evaluate(x, local_bandwidth)
        self.assertAlmostEqual(actual, expected)

    def test_alternative_implementation_single_pattern_l_neq_1(self):
        H = np.array([
            [0.080225998475784, 0.000182273891304],
            [0.000182273891304, 0.081385767033078]
        ])
        x = np.array([0.05, 0.05])
        expected = None
        local_bandwidth = 0.5
        actual = ShapeAdaptiveEpanechnikov(H, implementation=_ShapeAdaptiveEpanechnikov_Python).evaluate(x, local_bandwidth)
        self.assertAlmostEqual(actual, expected)

    def test_default_implementation_multiple_patterns_l_neq_1(self):
        H = np.array([
            [0.080225998475784, 0.000182273891304],
            [0.000182273891304, 0.081385767033078]
        ])
        x = np.array([
            [0.05, 0.05],
            [0.02, 0.03]
        ])
        local_bandwidth = np.array([0.5, 0.2])
        expected = np.array([
            None,
            None
        ])
        actual = ShapeAdaptiveEpanechnikov(H).evaluate(x, local_bandwidth)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_alternative_implementation_multiple_patterns_l_neq_1(self):
        H = np.array([
            [0.080225998475784, 0.000182273891304],
            [0.000182273891304, 0.081385767033078]
        ])
        x = np.array([
            [0.05, 0.05],
            [0.02, 0.03]
        ])
        local_bandwidth = np.array([0.5, 0.2])
        expected = np.array([
            None,
            None
        ])
        actual = ShapeAdaptiveEpanechnikov(H, implementation=_ShapeAdaptiveEpanechnikov_Python).evaluate(x, local_bandwidth)
        np.testing.assert_array_almost_equal(actual, expected)


class ShapeAdaptiveGaussianImpAbstractTest(object):

    def test_to_C_enum(self):
        actual = ShapeAdaptiveEpanechnikov.to_C_enum()
        expected = 3
        self.assertEqual(actual, expected)

    def test_evalute_single_pattern_l_eq_one(self):
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        x = np.array([0.05, 0.05, 0.05])
        actual = self._kernel_class(H).evaluate(x, 1.0)
        expected = None
        self.assertAlmostEqual(actual, expected)

    def test_evalute_single_pattern_l_neq_one(self):
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        x = np.array([0.05, 0.05, 0.05])
        local_bandwidth = 0.5
        actual = self._kernel_class(H).evaluate(x, 0.5)
        expected = None
        self.assertAlmostEqual(actual, expected)

    def test_evalute_multiple_patterns_l_eq_one(self):
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        x = np.array([
            [0.05, 0.05, 0.05],
            [0.02, 0.03, 0.04],
            [0.04, 0.05, 0.03]
        ])
        expected = np.array([
            None,
            None,
            None
        ])
        actual = self._kernel_class(H).evaluate(x)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evalute_multiple_patterns_l_neq_one(self):
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        local_bandwidths = np.array([0.5, 0.7, 0.2])
        x = np.array([
            [0.05, 0.05, 0.05],
            [0.02, 0.03, 0.04],
            [0.04, 0.05, 0.03]
        ])
        expected = np.array([
            None,
            None,
            None
        ])
        actual = self._kernel_class(H).evaluate(x, local_bandwidths=local_bandwidths)
        np.testing.assert_array_almost_equal(actual, expected)

class Test_ShapeAdaptiveEpanechnikov_Python(ShapeAdaptiveGaussianImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._kernel_class = _ShapeAdaptiveEpanechnikov_Python


class Test_ShapeAdaptiveEpanechnikov_C(ShapeAdaptiveGaussianImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._kernel_class = _ShapeAdaptiveEpanechnikov_C