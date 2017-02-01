from unittest import TestCase

import numpy as np

import kdeUtils.automaticWindowWidthMethods
from kde.kernels.epanechnikov import Epanechnikov
from kde.kernels.standardGaussian import StandardGaussian
from kde.kernels.testKernel import TestKernel
from kde.modifeidbreiman import _MBEEstimator_Python, _MBEEstimator_C, ModifiedBreimanEstimator
from kde.parzen import _ParzenEstimator_Python, _ParzenEstimator_C


class TestModifiedBreimanEstimator(TestCase):
    def estimate_test_helper(self, pilot_implementation, final_implementation):
        xi_s = np.array([[0, 0], [1, 1]])
        x_s = np.array([[0, 0]])
        pilot_kernel = TestKernel()
        final_kerel = TestKernel()
        number_of_grid_points = 2
        sensitivity = 0.5
        estimator = ModifiedBreimanEstimator(
            pilot_kernel=pilot_kernel, pilot_estimator_implementation=pilot_implementation,
            kernel=final_kerel, final_estimator_implementation=final_implementation,
            dimension=2, number_of_grid_points=number_of_grid_points,
            sensitivity=sensitivity,
            pilot_window_width_method=kdeUtils.automaticWindowWidthMethods.ferdosi
        )
        actual = estimator.estimate(xi_s=xi_s, x_s=x_s)
        expected = np.array([0.7708904])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_estimate_python_python(self):
        self.estimate_test_helper(_ParzenEstimator_Python, _MBEEstimator_Python)

    def test_estimate_python_C(self):
        self.estimate_test_helper(_ParzenEstimator_Python, _MBEEstimator_C)

    def test_estimate_C_python(self):
        self.estimate_test_helper(_ParzenEstimator_C, _MBEEstimator_Python)

    def test_estimate_C_C(self):
        self.estimate_test_helper(_ParzenEstimator_C, _MBEEstimator_C)

    def estimate_pilot_densities_test_helper(self, _parzen_implementation):
        xi_s = np.array([
            [0, 0],
            [1, 1]
        ])
        estimator = ModifiedBreimanEstimator(
            pilot_kernel=StandardGaussian(), pilot_estimator_implementation=_parzen_implementation,
            dimension=2, sensitivity=0.5
        )
        actual = estimator._estimate_pilot_densities(0.5, xi_s)
        expected = np.array([0.32413993511384709, 0.32413993511384709])
        np.testing.assert_array_almost_equal(actual, expected)

    def test__estimate_pilot_densities_python(self):
        self.estimate_pilot_densities_test_helper(_ParzenEstimator_Python)

    def test__estimate_pilot_densities_C(self):
        self.estimate_pilot_densities_test_helper(_ParzenEstimator_C)

    def test__compute_local_bandwidths(self):
        estimator = ModifiedBreimanEstimator(dimension=2, sensitivity=0.5)
        densities = np.array([1, 2, 3, 4, 5, 6])
        expected = np.array([1.730258699016973, 1.223477659281915, 0.998965325645141,
                             0.865129349508487, 0.773795213932460, 0.706375155933907])
        actual = estimator._compute_local_bandwidths(densities)
        np.testing.assert_array_almost_equal(expected, actual)


class ModifiedBreimanEstimatorImpAbstractTest(object):
    def setUp(self):
        super().setUp()
        self._estimator_class = None

    def test_estimate_epanechnikov(self):
        xi_s = np.array([[-1, -1], [1, 1], [0, 0]])
        x_s = np.array([[0, 0], [1, 1], [0, 1]])
        local_bandwidths = np.array([10, 20, 50])
        general_bandwidth = 0.5
        kernel = Epanechnikov()
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=kernel,
            local_bandwidths=local_bandwidths, general_bandwidth=general_bandwidth
        )
        actual = estimator.estimate()
        expected = np.array([0.01022836, 0.00823253, 0.00923044])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_estimate_gaussian(self):
        xi_s = np.array([[-1, -1], [1, 1], [0, 0]])
        x_s = np.array([[0, 0], [1, 1], [0, 1]])
        local_bandwidths = np.array([10, 20, 50])
        general_bandwidth = 0.5
        kernel = StandardGaussian()
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=kernel,
            local_bandwidths=local_bandwidths, general_bandwidth=general_bandwidth
        )
        actual = estimator.estimate()
        expected = np.array([0.00264898, 0.0024237, 0.00253281])
        np.testing.assert_array_almost_equal(actual, expected)


class Test_MBEEstimator_Python(ModifiedBreimanEstimatorImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
        self._estimator_class = _MBEEstimator_Python

    def test_estimate_pattern_gaussian(self):
        xi_s = np.array([[-1, -1], [1, 1], [0, 0]])
        x_s = np.array([[0, 0], [1, 1]])
        local_bandwidths = np.array([10, 20, 50])
        general_bandwidth = 0.5
        kernel = StandardGaussian()
        pattern = x_s[0]
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=kernel,
            local_bandwidths=local_bandwidths, general_bandwidth=general_bandwidth
        )
        actual = estimator._estimate_pattern(pattern)
        expected = 0.00264898 * 3
        self.assertAlmostEqual(actual, expected)

    def test_estimate_pattern_epanechnikov(self):
        xi_s = np.array([[-1, -1], [1, 1], [0, 0]])
        x_s = np.array([[0, 0], [1, 1]])
        local_bandwidths = np.array([10, 20, 50])
        general_bandwidth = 0.5
        kernel = Epanechnikov()
        pattern = x_s[0]
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=kernel,
            local_bandwidths=local_bandwidths, general_bandwidth=general_bandwidth
        )
        actual = estimator._estimate_pattern(pattern)
        expected = 0.01022836 * 3
        self.assertAlmostEqual(actual, expected)


class Test_MBEEstimator_C(ModifiedBreimanEstimatorImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
        self._estimator_class = _MBEEstimator_C
