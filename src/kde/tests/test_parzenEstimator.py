from unittest import TestCase

import numpy as np

from kde.kernels.standardGaussian import StandardGaussian
from kde.kernels.testKernel import TestKernel
from kde.parzen import _ParzenEstimator_C, _ParzenEstimator_Python, ParzenEstimator

class TestParzenEstimator(TestCase):
    def test_estimate_python(self):
        xi_s = np.array([[-1, -1], [0, 0], [1 / 2.0, 1 / 2.0]])
        x_s = np.array([[0, 0], [1 / 4.0, 1 / 2.0]])
        estimator = ParzenEstimator(dimension=2,
                                    bandwidth=0.25, kernel=StandardGaussian(),
                                    estimator_implementation=_ParzenEstimator_Python)
        actual = estimator.estimate(xi_s=xi_s, x_s=x_s)
        expected = np.array([0.0387795541707939, 0.0381443156873352])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_estimate_C(self):
        xi_s = np.array([[-1, -1], [0, 0], [1 / 2.0, 1 / 2.0]])
        x_s = np.array([[0, 0], [1 / 4.0, 1 / 2.0]])
        estimator = ParzenEstimator(dimension=2,
                                    bandwidth=0.25, kernel=StandardGaussian(),
                                    estimator_implementation=_ParzenEstimator_C)
        actual = estimator.estimate(xi_s=xi_s, x_s=x_s)
        expected = np.array([0.0387795541707939, 0.0381443156873352])
        np.testing.assert_array_almost_equal(actual, expected)


class ParzenEstimatorImpAbstractTest(object):
    def setUp(self):
        super().setUp()
        self._estimator_class = None

    def test_estimate(self):
        xi_s = np.array([[-1, -1], [0, 0], [1 / 2.0, 1 / 2.0]])
        x_s = np.array([[0, 0], [1 / 4.0, 1 / 2.0]])
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=StandardGaussian(), general_bandwidth=0.25)
        actual = estimator.estimate()
        expected = np.array([0.0387795541707939, 0.0381443156873352])
        np.testing.assert_array_almost_equal(actual, expected)


class Test_ParzenEstimator_Python(ParzenEstimatorImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
        self._estimator_class = _ParzenEstimator_Python

    def test_estimate_pattern(self):
        xi_s = np.array([[-1, -1], [0, 0], [1 / 2.0, 1 / 2.0]])
        x_s = np.array([[0, 0], [1 / 4.0, 1 / 2.0]])
        pattern = x_s[0]
        factor = 0.5
        estimator = _ParzenEstimator_Python(
            xi_s=xi_s, x_s=x_s,
            dimension=2,
            kernel=TestKernel(), general_bandwidth=0.75)
        actual = estimator._estimate_pattern(pattern=pattern, factor=factor)
        expected = 15 / 384.0
        self.assertAlmostEqual(actual, expected)


class Test_ParzenEstimator_C(ParzenEstimatorImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
        self._estimator_class = _ParzenEstimator_C
