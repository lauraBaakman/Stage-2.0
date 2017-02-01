from unittest import TestCase

import numpy as np

from kde.kernels.epanechnikov import Epanechnikov
from kde.kernels.standardGaussian import StandardGaussian
from kde.modifeidbreiman import _MBEEstimator_Python, _MBEEstimator_C, ModifiedBreimanEstimator
from kde.parzen import _ParzenEstimator_Python, _ParzenEstimator_C


class TestModifiedBreimanEstimator(TestCase):
    def estimate_test_helper(self, pilot_implementation, final_implementation):
        xi_s = np.array([])
        x_s = np.array([])
        pilot_kernel = Epanechnikov
        final_kerel = StandardGaussian
        number_of_grid_points = 2
        sensitivity = 0.5
        estimator = ModifiedBreimanEstimator(
            pilot_kernel=pilot_kernel, pilot_estimator_implementation=pilot_implementation,
            kernel=final_kerel, final_estimator_implementation=final_implementation,
            dimension=2, number_of_grid_points=number_of_grid_points,
            sensitivity=sensitivity
        )
        actual = estimator.estimate(xi_s=xi_s, x_s=x_s)
        expected = np.array([])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_estimate_python_python(self):
        self.estimate_test_helper(_ParzenEstimator_Python, _MBEEstimator_Python)

    def test_estimate_python_C(self):
        self.estimate_test_helper(_ParzenEstimator_Python, _MBEEstimator_C)

    def test_estimate_C_python(self):
        self.estimate_test_helper(_ParzenEstimator_C, _MBEEstimator_Python)

    def test_estimate_C_C(self):
        self.estimate_test_helper(_ParzenEstimator_C, _ParzenEstimator_C)

    def estimate_pilot_densities_gaussian_test_helper(self, _parzen_implementation):
        xi_s = np.array([
            [0, 0],
            [1, 1]
        ])
        estimator = ModifiedBreimanEstimator(
            pilot_kernel=StandardGaussian(), pilot_estimator_implementation=_parzen_implementation,
            dimension=2, sensitivity=0.5
        )
        actual = estimator._estimate_pilot_densities(0.5, xi_s)
        expected = np.array([])
        np.testing.assert_array_almost_equal(actual, expected)

    def test__estimate_pilot_densities_python_gaussian(self):
        self.estimate_pilot_densities_gaussian_test_helper(_ParzenEstimator_Python)

    def test__estimate_pilot_densities_C_gaussian(self):
        self.estimate_pilot_densities_gaussian_test_helper(_ParzenEstimator_C)

    def estimate_pilot_densities_epanechnikov_test_helper(self, _parzen_implementation):
        xi_s = np.array([
            [0, 0],
            [1, 1]
        ])
        estimator = ModifiedBreimanEstimator(
            pilot_kernel=Epanechnikov(), pilot_estimator_implementation=_parzen_implementation,
            dimension=2, sensitivity=0.5
        )
        actual = estimator._estimate_pilot_densities(0.5, xi_s)
        expected = np.array([])
        np.testing.assert_array_almost_equal(actual, expected)

    def test__estimate_pilot_densities_python_epanechnikov(self):
        self.estimate_pilot_densities_epanechnikov_test_helper(_ParzenEstimator_Python)

    def test__estimate_pilot_densities_C_epanechnikov(self):
        self.estimate_pilot_densities_epanechnikov_test_helper(_ParzenEstimator_C)

    def test__compute_local_bandwidths(self):
        densities = np.array([1, 2, 3, 4, 5, 6])
        expected = np.array([1.730258699016973, 1.223477659281915, 0.998965325645141,
                             0.865129349508487, 0.773795213932460, 0.706375155933907])
        actual = self.estimator._compute_local_bandwidths(densities)
        np.testing.assert_array_almost_equal(expected, actual)


class ModifiedBreimanEstimatorImpAbstractTest(object):
    def setUp(self):
        super().setUp()
        self._estimator_class = None

    def test_estimate_epanechnikov(self):
        xi_s = np.array([[-1, -1], [0, 0], [1 / 2.0, 1 / 2.0]])
        local_bandwidths = np.array([])
        x_s = np.array([[0, 0], [1 / 4.0, 1 / 2.0]])
        general_bandwidth = None
        kernel = Epanechnikov()
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=kernel,
            local_bandwidths=local_bandwidths, general_bandwidth=general_bandwidth
        )
        actual = estimator.estimate()
        expected = np.array([])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_estimate_gaussian(self):
        xi_s = np.array([[-1, -1], [0, 0], [1 / 2.0, 1 / 2.0]])
        local_bandwidths = np.array([])
        x_s = np.array([[0, 0], [1 / 4.0, 1 / 2.0]])
        general_bandwidth = None
        kernel = StandardGaussian()
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=kernel,
            local_bandwidths=local_bandwidths, general_bandwidth=general_bandwidth
        )
        actual = estimator.estimate()
        expected = np.array([])
        np.testing.assert_array_almost_equal(actual, expected)


class Test_MBEEstimator_Python(ModifiedBreimanEstimatorImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
        self._estimator_class = _MBEEstimator_Python

    def test_estimate_pattern_gaussian(self):
        xi_s = np.array([[-1, -1], [0, 0], [1 / 2.0, 1 / 2.0]])
        local_bandwidths = np.array([])
        x_s = np.array([[0, 0], [1 / 4.0, 1 / 2.0]])
        pattern = x_s[0]
        general_bandwidth = None
        kernel = StandardGaussian()
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=kernel,
            local_bandwidths=local_bandwidths, general_bandwidth=general_bandwidth
        )
        actual = estimator._estimate_pattern(pattern)
        expected = None
        self.assertAlmostEqual(actual, expected)

    def test_estimate_pattern_epanechnikov(self):
        xi_s = np.array([[-1, -1], [0, 0], [1 / 2.0, 1 / 2.0]])
        local_bandwidths = np.array([])
        x_s = np.array([[0, 0], [1 / 4.0, 1 / 2.0]])
        pattern = x_s[0]
        general_bandwidth = None
        kernel = Epanechnikov()
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=kernel,
            local_bandwidths=local_bandwidths, general_bandwidth=general_bandwidth
        )
        actual = estimator._estimate_pattern(pattern)
        expected = None
        self.assertAlmostEqual(actual, expected)


class Test_MBEEstimator_C(ModifiedBreimanEstimatorImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
        self._estimator_class = _MBEEstimator_C
