from unittest import TestCase, skip

import numpy as np

from kde.kernels.kernel import KernelException
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

    def test_evaluate_2(self):
        # Single pattern, local bandwidth \neq 1
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        bandwidth = 0.5
        x = np.array([0.05, 0.05, 0.05])
        expected = 0.121703390601269
        actual = self._kernel_class(H).evaluate(x, local_bandwidths=bandwidth)
        self.assertAlmostEqual(expected, actual)

    def test_evaluate_3(self):
        # Multiple patterns, local bandwidth \neq 1
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
            0.121703390601269,
            0.045915970935366,
            1.656546521485471
        ])
        actual = self._kernel_class(H).evaluate(x, local_bandwidths=local_bandwidths)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_4(self):
        # Single pattern, local bandwidth \neq 1
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        bandwidth = np.array([0.5])
        x = np.array([0.05, 0.05, 0.05])
        expected = 0.121703390601269
        actual = self._kernel_class(H).evaluate(x, local_bandwidths=bandwidth)
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_5(self):
        # Single pattern, local bandwidth \neq 1
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        bandwidth = 0.5
        x = np.array([0.02, 0.03, 0.04])
        expected = 0.125046649030584
        actual = self._kernel_class(H).evaluate(x, local_bandwidths=bandwidth)
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_6(self):
        # Single pattern, local bandwidth \neq 1
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        bandwidth = 0.7
        x = np.array([0.02, 0.03, 0.04])
        expected = 0.045915970935366
        actual = self._kernel_class(H).evaluate(x, local_bandwidths=bandwidth)
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_7(self):
        # Single pattern, local bandwidth \neq 1
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        bandwidth = 0.2
        x = np.array([0.04, 0.05, 0.03])
        expected = 1.656546521485471
        actual = self._kernel_class(H).evaluate(x, local_bandwidths=bandwidth)
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_8(self):
        # Multiple patterns, local bandwidth \neq 1
        H = np.array([[+0.832940370216, -0.416470185108],
                      [-0.416470185108, +0.832940370216]])
        bandwidth = np.array([0.840896194314, 1.18920742746, 1.18920742746, 0.840896194314])
        x = np.array([[0.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 0.0],
                      [1.0, 1.0]])
        expected = np.array([
            0.4325599148816852,
            0.06969541947584654,
            0.06969541947584654,
            0.00012445121880528157
        ])
        actual = self._kernel_class(H).evaluate(x, local_bandwidths=bandwidth)
        np.testing.assert_array_almost_equal(actual, expected)


class ShapeAdaptiveGaussian_Python(ShapeAdaptiveGaussianImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._kernel_class = _ShapeAdaptiveGaussian_Python

    def test_evaluate_pattern_0(self):
        H = np.array([[4, 2],
                      [7, 6]])
        x = np.array([0.05, 0.05])
        local_bandwidth = 1.0
        expected = 0.015914499621880
        actual = self._kernel_class(H)._evaluate_pattern(x, local_bandwidth)
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_pattern_1(self):
        H = np.array([[4, 2],
                      [7, 6]])
        local_bandwidth = 0.5
        x = np.array([0.05, 0.05])
        expected = 0.063646063731720
        actual = self._kernel_class(H)._evaluate_pattern(x, local_bandwidth)
        self.assertAlmostEqual(actual, expected)


class ShapeAdaptiveGaussian_C(ShapeAdaptiveGaussianImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._kernel_class = _ShapeAdaptiveGaussian_C
