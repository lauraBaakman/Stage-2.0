from __future__ import division

from unittest import TestCase

import numpy as np

import kde.utils.automaticWindowWidthMethods as automaticWindowWidthMethods
from kde.kernels.gaussian import Gaussian
from kde.kernels.testKernel import TestKernel
from kde.kernels.epanechnikov import Epanechnikov
from kde.parzen import _ParzenEstimator_C, _ParzenEstimator_Python, ParzenEstimator
import kde.utils._utils as _utils
from kde.utils.grid import Grid

number_of_threads = 1


class TestParzenEstimator(TestCase):
    def setUp(self):
        _utils.set_num_threads(number_of_threads)

    def tearDown(self):
        _utils.reset_num_threads()

    def test_estimate_python(self):
        xi_s = np.array([[-1, -1], [0, 0], [1 / 2.0, 1 / 2.0]])
        x_s = np.array([[0, 0], [1 / 4.0, 1 / 2.0]])
        estimator = ParzenEstimator(dimension=2,
                                    bandwidth=4, kernel_class=Gaussian,
                                    estimator_implementation=_ParzenEstimator_Python)
        actual = estimator.estimate(xi_s=xi_s, x_s=x_s).densities
        expected = np.array([0.0096947375, 0.0095360625])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_estimate_python_dont_pass_bandwidth(self):
        xi_s = np.array([[-1, -1], [0, 0], [1 / 2.0, 1 / 2.0]])
        x_s = np.array([[0, 0], [1 / 4.0, 1 / 2.0]])
        estimator = ParzenEstimator(dimension=2, kernel_class=Gaussian,
                                    estimator_implementation=_ParzenEstimator_Python)
        actual = estimator.estimate(xi_s=xi_s, x_s=x_s).densities
        expected = np.array([0.15133033, 0.14270123])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_estimate_python_pass_bandwidth_with_estimate(self):
        xi_s = np.array([[-1, -1], [0, 0], [1 / 2.0, 1 / 2.0]])
        x_s = np.array([[0, 0], [1 / 4.0, 1 / 2.0]])
        estimator = ParzenEstimator(dimension=2, kernel_class=Gaussian,
                                    estimator_implementation=_ParzenEstimator_Python)
        actual = estimator.estimate(xi_s=xi_s, x_s=x_s, general_bandwidth=4).densities
        expected = np.array([0.0096947375, 0.0095360625])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_estimate_C(self):
        xi_s = np.array([[-1, -1], [0, 0], [1 / 2.0, 1 / 2.0]])
        x_s = np.array([[0, 0], [1 / 4.0, 1 / 2.0]])
        estimator = ParzenEstimator(dimension=2,
                                    bandwidth=4, kernel_class=Gaussian,
                                    estimator_implementation=_ParzenEstimator_C)
        actual = estimator.estimate(xi_s=xi_s, x_s=x_s).densities
        expected = np.array([0.0096947375, 0.0095360625])
        np.testing.assert_array_almost_equal(actual, expected)


class ParzenEstimatorImpAbstractTest(object):
    def setUp(self):
        super(ParzenEstimatorImpAbstractTest, self).setUp()
        self._estimator_class = None

    def test_estimate_standard_gaussian(self):
        xi_s = np.array([[-1, -1], [0, 0], [1 / 2.0, 1 / 2.0]])
        x_s = np.array([[0, 0], [1 / 4.0, 1 / 2.0]])
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=Gaussian(), general_bandwidth=4)
        actual = estimator.estimate().densities
        expected = np.array([0.0096947375, 0.0095360625])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_estimate_test_kernel(self):
        xi_s = np.array([[-1, -1], [0, 0], [1 / 2.0, 1 / 2.0]])
        x_s = np.array([[0, 0], [1 / 4.0, 1 / 2.0]])
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=TestKernel(), general_bandwidth=4)
        actual = estimator.estimate().densities
        expected = np.array([3 / 384.0, 15 / 1536.0])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_estimate_epanechnikov(self):
        xi_s = Grid.cover(
            points=np.array([[0, 0], [3, 3]]),
            number_of_grid_points=4
        ).grid_points
        x_s = np.array([
            [0.5, 0.7],
            [2.3, 1.8],
            [0.7, 2.8],
            [0.1, 2.3],
            [2.7, 1.6],
            [1.3, 0.9]
        ])
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=Epanechnikov(), general_bandwidth=4)
        actual = estimator.estimate().densities
        expected = np.array([
            0.007545933739,
            0.007636453113,
            0.00747729817,
            0.007450440773,
            0.007564833389,
            0.00766927882
        ])
        np.testing.assert_array_almost_equal(actual, expected)


class Test_ParzenEstimator_Python(ParzenEstimatorImpAbstractTest, TestCase):
    def setUp(self):
        super(Test_ParzenEstimator_Python, self).setUp()
        self._estimator_class = _ParzenEstimator_Python
        _utils.set_num_threads(number_of_threads)

    def tearDown(self):
        _utils.reset_num_threads()

    def test_estimate_pattern(self):
        xi_s = np.array([[-1, -1], [0, 0], [1 / 2.0, 1 / 2.0]])
        x_s = np.array([[0, 0], [1 / 4.0, 1 / 2.0]])
        pattern = x_s[1]
        factor = 1 / 48
        estimator = _ParzenEstimator_Python(
            xi_s=xi_s, x_s=x_s,
            dimension=2,
            kernel=TestKernel(), general_bandwidth=4)
        actual = estimator._estimate_pattern(
            pattern=pattern, factor=factor
        ).densities
        expected = 15 / 1536.0
        self.assertAlmostEqual(actual, expected)


class Test_ParzenEstimator_C(ParzenEstimatorImpAbstractTest, TestCase):
    def setUp(self):
        super(Test_ParzenEstimator_C, self).setUp()
        self._estimator_class = _ParzenEstimator_C
        _utils.set_num_threads(number_of_threads)

    def tearDown(self):
        _utils.reset_num_threads()
