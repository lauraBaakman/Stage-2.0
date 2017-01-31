from unittest import TestCase

import numpy as np

import kde
import kde.modifeidbreiman as mbe
import kdeUtils.automaticWindowWidthMethods
from kde.kernels.testKernel import TestKernel
from kde.kernels.epanechnikov import Epanechnikov
from kdeUtils.grid import Grid
from kde.modifeidbreiman import _MBEEstimator, _MBEEstimator_Python, ModifiedBreimanEstimator


class TestModifiedBreimanEstimator(TestCase):

    def setUp(self):
        super().setUp()
        self.estimator = kde.ModifiedBreimanEstimator(dimension=2, sensitivity=0.5)

    def test__compute_local_bandwidths(self):
        densities = np.array([1, 2, 3, 4, 5, 6])
        expected = np.array([1.730258699016973, 1.223477659281915, 0.998965325645141,
                             0.865129349508487, 0.773795213932460, 0.706375155933907])
        actual = self.estimator._compute_local_bandwidths(densities)
        np.testing.assert_array_almost_equal(expected, actual)

    def test_estimate_on_grid(self):
        xi_s = Grid((0, 3), number_of_grid_points=2).grid_points
        x_s = xi_s
        dimension = 1
        kernel = TestKernel()
        local_bandwidths = np.array([1, 2])
        general_bandwidth = 0.5
        estimator = mbe._MBEEstimator_Python(xi_s=xi_s, x_s=x_s,
                                             dimension=dimension, kernel=kernel,
                                             local_bandwidths=local_bandwidths, general_bandwidth=general_bandwidth)
        actual = estimator.estimate()
        expected = np.array([1.5, 6])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_estimate_with_xs(self):
        xi_s = Grid((0, 3), number_of_grid_points=2).grid_points
        x_s = np.array([[2]])
        dimension = 1
        kernel = TestKernel()
        local_bandwidths = np.array([1, 2])
        general_bandwidth = 0.5
        estimator = mbe._MBEEstimator_Python(xi_s=xi_s, x_s=x_s,
                                             dimension=dimension, kernel=kernel,
                                             local_bandwidths=local_bandwidths, general_bandwidth=general_bandwidth)
        actual = estimator.estimate()
        expected = np.array([4.5])
        np.testing.assert_array_almost_equal(actual, expected)


class MBEEstimatorAbstractTest(TestCase):

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


class Test_MBEEstimator_Python(MBEEstimatorAbstractTest):

    def setUp(self):
        super().setUp()
        self._estimator_class =_MBEEstimator_Python

    def test_estimate_pattern(self):
        raise NotImplementedError()


class Test_MBEEstimator(MBEEstimatorAbstractTest):

    def setUp(self):
        super().setUp()
        self._estimator_class = _MBEEstimator