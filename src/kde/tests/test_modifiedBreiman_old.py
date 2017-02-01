from unittest import TestCase

import numpy as np

import kdeUtils.automaticWindowWidthMethods
from kde.kernels.epanechnikov import Epanechnikov
from kde.kernels.standardGaussian import StandardGaussian
from kde.kernels.testKernel import TestKernel
from kde.modifeidbreiman import ModifiedBreimanEstimator, _MBEEstimator_C, _MBEEstimator_Python

class TestModifiedBreimanEstimator(TestCase):

    def setUp(self):
        super().setUp()
        self.estimator = ModifiedBreimanEstimator(dimension=2, sensitivity=0.5)

    def test__compute_local_bandwidths(self):
        densities = np.array([1, 2, 3, 4, 5, 6])
        expected = np.array([1.730258699016973, 1.223477659281915, 0.998965325645141,
                             0.865129349508487, 0.773795213932460, 0.706375155933907])
        actual = self.estimator._compute_local_bandwidths(densities)
        np.testing.assert_array_almost_equal(expected, actual)

    def test_estimate_python_python(self):
        estimator = ModifiedBreimanEstimator(
            dimension=2, sensitivity=0.5, number_of_grid_points=2,
            pilot_kernel=TestKernel(), kernel=TestKernel(),
            pilot_window_width_method=kdeUtils.automaticWindowWidthMethods.test,
            pilot_estimator_implementation=_ParzenEstimator_Python,
            final_estimator_implementation=_MBEEstimator_Python)
        xi_s = np.array([
            [0, 0],
            [1, 1]
        ])
        x_s = np.array([
            [0, 0],
        ])
        actual = estimator.estimate(xi_s=xi_s, x_s=x_s)
        expected = np.array([4])
        np.testing.assert_almost_equal(actual, expected)

    def test_estimate_python_C(self):
        estimator = ModifiedBreimanEstimator(
            dimension=2, sensitivity=0.5, number_of_grid_points=2,
            pilot_kernel=TestKernel(), kernel=TestKernel(),
            pilot_window_width_method=kdeUtils.automaticWindowWidthMethods.test,
            pilot_estimator_implementation=_ParzenEstimator_Python,
            final_estimator_implementation=_MBEEstimator_C)
        xi_s = np.array([
            [0, 0],
            [1, 1]
        ])
        x_s = np.array([
            [0, 0],
        ])
        actual = estimator.estimate(xi_s=xi_s, x_s=x_s)
        expected = np.array([4])
        np.testing.assert_almost_equal(actual, expected)

    def test_estimate_C_python(self):
        estimator = ModifiedBreimanEstimator(
            dimension=2, sensitivity=0.5, number_of_grid_points=2,
            pilot_kernel=TestKernel(), kernel=TestKernel(),
            pilot_window_width_method=kdeUtils.automaticWindowWidthMethods.test,
            pilot_estimator_implementation=_ParzenEstimator,
            final_estimator_implementation=_MBEEstimator_Python)
        xi_s = np.array([
            [0, 0],
            [1, 1]
        ])
        x_s = np.array([
            [0, 0],
        ])
        actual = estimator.estimate(xi_s=xi_s, x_s=x_s)
        expected = np.array([4])
        np.testing.assert_almost_equal(actual, expected)

    def test_estimate_C_C(self):
        estimator = ModifiedBreimanEstimator(
            dimension=2, sensitivity=0.5, number_of_grid_points=2,
            pilot_kernel=TestKernel(), kernel=TestKernel(),
            pilot_window_width_method=kdeUtils.automaticWindowWidthMethods.test,
            pilot_estimator_implementation=_ParzenEstimator,
            final_estimator_implementation=_MBEEstimator_C)
        xi_s = np.array([
            [0, 0],
            [1, 1]
        ])
        x_s = np.array([
            [0, 0],
        ])
        actual = estimator.estimate(xi_s=xi_s, x_s=x_s)
        expected = np.array([4])
        np.testing.assert_almost_equal(actual, expected)

    def test_estimate_pilot_densitites_python(self):
        estimator = ModifiedBreimanEstimator(
            dimension=2, sensitivity=0.5, number_of_grid_points=2,
            pilot_kernel=StandardGaussian, kernel=TestKernel(),
            pilot_window_width_method=kdeUtils.automaticWindowWidthMethods.test,
            pilot_estimator_implementation=_ParzenEstimator_Python)
        xi_s = np.array([
            [0, 0],
            [1, 1]
        ])
        actual = estimator._estimate_pilot_densities(0.5, xi_s)
        expected = np.array([0.2820062130103994, 0.2820062130103994])
        np.testing.assert_almost_equal(actual, expected)

    def test_estimate_pilot_densitites_C(self):
        estimator = ModifiedBreimanEstimator(
            dimension=2, sensitivity=0.5, number_of_grid_points=2,
            pilot_kernel=StandardGaussian, kernel=TestKernel(),
            pilot_window_width_method=kdeUtils.automaticWindowWidthMethods.test,
            pilot_estimator_implementation=_ParzenEstimator)
        xi_s = np.array([
            [0, 0],
            [1, 1]
        ])
        actual = estimator._estimate_pilot_densities(0.5, xi_s)
        expected = np.array([0.2820062130103994, 0.2820062130103994])
        np.testing.assert_almost_equal(actual, expected)


class MBEEstimatorAbstractTest(object):

    def setUp(self):
        super().setUp()
        self._estimator_class = None

    def test_estimate_1(self):
        xi_s = np.array([[-1, -1], [1, 1], [0, 0]])
        x_s = np.array([[0, 0], [1, 1]])
        local_bandwidths = np.array([10, 20, 50])
        estimator = self._estimator_class(xi_s= xi_s, x_s=x_s, dimension=2,
                                          kernel=Epanechnikov(),
                                          local_bandwidths=local_bandwidths, general_bandwidth=0.5)
        actual = estimator.estimate()
        expected = np.array([0.010228357933333333, 0.00823253])
        np.testing.assert_array_almost_equal(actual, expected)


class Test_MBEEstimator_Python(MBEEstimatorAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._estimator_class = _MBEEstimator_Python

    def test_estimate_pattern_1(self):
        xi_s = np.array([[-1, -1], [1, 1], [0, 0]])
        x_s = np.array([[0, 0], [1, 1]])
        local_bandwidths = np.array([10, 20, 50])
        estimator = self._estimator_class(xi_s=xi_s, x_s=x_s, dimension=2,
                                          kernel=Epanechnikov(),
                                          local_bandwidths=local_bandwidths, general_bandwidth=0.5)
        actual = estimator._estimate_pattern(np.array([0, 0]))
        expected = 0.010228357933333333 * 3
        self.assertAlmostEqual(actual, expected)

    def test_estimate_pattern_2(self):
        xi_s = np.array([[-1, -1], [1, 1], [0, 0]])
        x_s = np.array([[0, 0], [1, 1]])
        local_bandwidths = np.array([10, 20, 50])
        estimator = self._estimator_class(xi_s=xi_s, x_s=x_s, dimension=2,
                                          kernel=Epanechnikov(),
                                          local_bandwidths=local_bandwidths, general_bandwidth=0.5)
        actual = estimator._estimate_pattern(np.array([1, 1]))
        expected = 0.00823253 * 3
        self.assertAlmostEqual(actual, expected)


class Test_MBEEstimator(MBEEstimatorAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._estimator_class = _MBEEstimator_C
