from unittest import TestCase, skip

import numpy as np

from kde.kernels.epanechnikov import Epanechnikov
from kde.kernels.standardGaussian import StandardGaussian
from kde.kernels.testKernel import TestKernel
from kde.parzen import _ParzenEstimator_C, _ParzenEstimator_Python, ParzenEstimator

class TestParzenEstimator(TestCase):
    def test_estimate_python(self):
        xi_s = np.array([[-1, -1], [0, 0], [1 / 2.0, 1 / 2.0]])
        x_s = np.array([[0, 0], [1 / 4.0, 1 / 2.0]])
        estimator = ParzenEstimator(dimension=2,
                                    bandwidth=4, kernel_class=StandardGaussian,
                                    estimator_implementation=_ParzenEstimator_Python)
        actual = estimator.estimate(xi_s=xi_s, x_s=x_s)
        expected = np.array([0.0096947375, 0.0095360625])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_estimate_C(self):
        xi_s = np.array([[-1, -1], [0, 0], [1 / 2.0, 1 / 2.0]])
        x_s = np.array([[0, 0], [1 / 4.0, 1 / 2.0]])
        estimator = ParzenEstimator(dimension=2,
                                    bandwidth=4, kernel_class=StandardGaussian,
                                    estimator_implementation=_ParzenEstimator_C)
        actual = estimator.estimate(xi_s=xi_s, x_s=x_s)
        expected = np.array([0.0096947375, 0.0095360625])
        np.testing.assert_array_almost_equal(actual, expected)


class ParzenEstimatorImpAbstractTest(object):
    def setUp(self):
        super().setUp()
        self._estimator_class = None

    def test_estimate_standard_gaussian(self):
        xi_s = np.array([[-1, -1], [0, 0], [1 / 2.0, 1 / 2.0]])
        x_s = np.array([[0, 0], [1 / 4.0, 1 / 2.0]])
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=StandardGaussian(), general_bandwidth=4)
        actual = estimator.estimate()
        expected = np.array([0.0096947375, 0.0095360625])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_estimate_test_kernel(self):
        xi_s = np.array([[-1, -1], [0, 0], [1 / 2.0, 1 / 2.0]])
        x_s = np.array([[0, 0], [1 / 4.0, 1 / 2.0]])
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=TestKernel(), general_bandwidth=4)
        actual = estimator.estimate()
        expected = np.array([3 / 384.0, 15 / 1536.0])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_estimate_epanechnikov(self):
        xi_s = np.array([[-1, -1], [0, 0], [1 / 2.0, 1 / 2.0]])
        x_s = np.array([[0, 0], [1 / 4.0, 1 / 2.0]])
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=Epanechnikov(), general_bandwidth=4)
        actual = estimator.estimate()
        expected = np.array([0.048652803495835, 0.046243112756654])
        np.testing.assert_array_almost_equal(actual, expected)


class Test_ParzenEstimator_Python(ParzenEstimatorImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
        self._estimator_class = _ParzenEstimator_Python

    def test_estimate_pattern(self):
        xi_s = np.array([[-1, -1], [0, 0], [1 / 2.0, 1 / 2.0]])
        x_s = np.array([[0, 0], [1 / 4.0, 1 / 2.0]])
        pattern = x_s[1]
        factor = 1 / 48
        estimator = _ParzenEstimator_Python(
            xi_s=xi_s, x_s=x_s,
            dimension=2,
            kernel=TestKernel(), general_bandwidth=4)
        actual = estimator._estimate_pattern(pattern=pattern, factor=factor)
        expected = 15 / 1536.0
        self.assertAlmostEqual(actual, expected)


class Test_ParzenEstimator_C(ParzenEstimatorImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
        self._estimator_class = _ParzenEstimator_C
