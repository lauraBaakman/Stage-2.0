from unittest import TestCase

import numpy as np

from kde.kernels.shapeadaptivegaussian import \
    ShapeAdaptiveGaussian, \
    _ShapeAdaptiveGaussian_C, \
    _ShapeAdaptiveGaussian_Python


class TestShapeAdaptiveGaussian(TestCase):

    def test_to_C_enum(self):
        actual = ShapeAdaptiveGaussian.to_C_enum()
        expected = 4
        self.assertEqual(actual, expected)

    def test_default_implementation(self):
        H = np.array([
            [0.080225998475784, 0.000182273891304],
            [0.000182273891304, 0.081385767033078]
                ])
        x = np.array([0.05, 0.05])
        expected = 16.649507462597573
        actual = ShapeAdaptiveGaussian(H).evaluate(x)
        self.assertAlmostEqual(actual, expected)

    def test_alternative_implementation(self):
        H = np.array([
            [0.080225998475784, 0.000182273891304],
            [0.000182273891304, 0.081385767033078]
                ])
        x = np.array([0.05, 0.05])
        expected = 16.649507462597573
        actual = ShapeAdaptiveGaussian(H, _ShapeAdaptiveGaussian_C).evaluate(x)
        self.assertAlmostEqual(actual, expected)


class ShapeAdaptiveGaussianImpAbstractTest(object):

    def test_evaluate_0(self):
        # Single pattern, local bandwidth = 1
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        x = np.array([0.05, 0.05, 0.05])
        expected = 0.015705646827791
        actual = self._kernel_class(H).evaluate(x)
        self.assertAlmostEqual(expected, actual)

    def test_evaluate_1(self):
        # Multiple patterns, local bandwidth = 1
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        x = np.array([
            [0.05, 0.05, 0.05],
            [0.02, 0.03, 0.04],
            [0.04, 0.05, 0.03]
        ])
        expected = np.array([
            0.015705646827791,
            0.015812413849947,
            0.015759235403527
        ])
        actual = self._kernel_class(H).evaluate(x)
        np.testing.assert_array_almost_equal(actual, expected)


class ShapeAdaptiveGaussian_Python(ShapeAdaptiveGaussianImpAbstractTest, TestCase):

    def setUp(self):
        super(ShapeAdaptiveGaussian_Python, self).setUp()
        self._kernel_class = _ShapeAdaptiveGaussian_Python

    def test_evaluate_pattern_0(self):
        H = np.array([[4, 2],
                      [7, 6]])
        x = np.array([0.05, 0.05])
        local_bandwidth = 1.0
        expected = 0.015914499621880
        actual = self._kernel_class(H)._evaluate_pattern(x, local_bandwidth)
        self.assertAlmostEqual(actual, expected)


class ShapeAdaptiveGaussian_C(ShapeAdaptiveGaussianImpAbstractTest, TestCase):

    def setUp(self):
        super(ShapeAdaptiveGaussian_C, self).setUp()
        self._kernel_class = _ShapeAdaptiveGaussian_C
