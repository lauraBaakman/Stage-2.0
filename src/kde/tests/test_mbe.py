from unittest import TestCase

import numpy as np
import warnings

import kde
from kde.kernels.gaussian import Gaussian
from kde.kernels.testKernel import TestKernel
from kde.mbe import _MBEEstimator_Python, _MBEEstimator_C, MBEstimator
from kde.parzen import _ParzenEstimator_Python, _ParzenEstimator_C
import kde.utils._utils as _utils


class TestModifiedBreimanEstimator(TestCase):
    def setUp(self):
        _utils.set_num_threads(2)
        warnings.simplefilter("always")

    def tearDown(self):
        _utils.reset_num_threads()

    def estimate_test_helper(self, pilot_implementation, final_implementation):
        xi_s = np.array([[0, 0], [1, 1]])
        x_s = np.array([[0, 0]])
        pilot_kernel = TestKernel
        final_kerel = TestKernel
        number_of_grid_points = 2
        sensitivity = 0.5
        estimator = MBEstimator(
            pilot_kernel_class=pilot_kernel, pilot_estimator_implementation=pilot_implementation,
            kernel_class=final_kerel, final_estimator_implementation=final_implementation,
            dimension=2, number_of_grid_points=number_of_grid_points,
            sensitivity=sensitivity,
            pilot_window_width_method=kde.utils.automaticWindowWidthMethods.ferdosi
        )
        actual = estimator.estimate(xi_s=xi_s, x_s=x_s)

        expected_densities = np.array([0.7708904])
        expected_num_patterns_used_for_density = np.array([1])
        expected_xis = xi_s

        np.testing.assert_array_almost_equal(actual.densities, expected_densities)
        np.testing.assert_array_almost_equal(actual.xis, expected_xis)
        np.testing.assert_array_almost_equal(
            actual.num_used_patterns,
            expected_num_patterns_used_for_density
        )

    def test_estimate_python_python(self):
        self.estimate_test_helper(_ParzenEstimator_Python, _MBEEstimator_Python)

    def test_estimate_python_C(self):
        self.estimate_test_helper(_ParzenEstimator_Python, _MBEEstimator_C)

    def test_estimate_C_python(self):
        self.estimate_test_helper(_ParzenEstimator_C, _MBEEstimator_Python)

    def test_estimate_C_C(self):
        self.estimate_test_helper(_ParzenEstimator_C, _MBEEstimator_C)

    def test_estimate_C_C_pass_pilot_densities_and_general_bandwidth(self):
        with warnings.catch_warnings(record=True):
            xi_s = np.array([[0, 0], [1, 1]])
            x_s = np.array([[0, 0]])
            pilot_densities = np.array([0.7708904,  0.7708904])
            general_bandwidth = 0.86561702453337819
            pilot_kernel = TestKernel
            final_kerel = TestKernel
            number_of_grid_points = 2
            sensitivity = 0.5
            estimator = MBEstimator(
                pilot_kernel_class=pilot_kernel, pilot_estimator_implementation=_ParzenEstimator_C,
                kernel_class=final_kerel, final_estimator_implementation=_MBEEstimator_C,
                dimension=2, number_of_grid_points=number_of_grid_points,
                sensitivity=sensitivity,
                pilot_window_width_method=kde.utils.automaticWindowWidthMethods.ferdosi
            )
            actual = estimator.estimate(
                xi_s=xi_s, x_s=x_s,
                pilot_densities=pilot_densities, general_bandwidth=general_bandwidth
            )

            expected_densities = np.array([0.7708904])
            expected_num_patterns_used_for_density = np.array([1])
            expected_xis = xi_s

            np.testing.assert_array_almost_equal(actual.densities, expected_densities)
            np.testing.assert_array_almost_equal(actual.xis, expected_xis)
            np.testing.assert_array_almost_equal(
                actual.num_used_patterns,
                expected_num_patterns_used_for_density
            )

    def estimate_pilot_densities_test_helper(self, _parzen_implementation):
        xi_s = np.array([
            [0, 0],
            [1, 1]
        ])
        estimator = MBEstimator(
            pilot_kernel_class=Gaussian, pilot_estimator_implementation=_parzen_implementation,
            dimension=2, sensitivity=0.5
        )
        actual = estimator._estimate_pilot_densities(0.5, xi_s)
        expected = np.array([0.32413993511384709, 0.32413993511384709])
        np.testing.assert_array_almost_equal(actual, expected)

    def test__estimate_pilot_densities_python(self):
        self.estimate_pilot_densities_test_helper(_ParzenEstimator_Python)

    def test__estimate_pilot_densities_C(self):
        self.estimate_pilot_densities_test_helper(_ParzenEstimator_C)

    def test__compute_local_bandwidths_no_non_zero_densities(self):
        estimator = MBEstimator(dimension=2, sensitivity=0.5)
        densities = np.array([1, 2, 3, 4, 5, 6])
        expected = np.array([1.730258699016973, 1.223477659281915, 0.998965325645141,
                             0.865129349508487, 0.773795213932460, 0.706375155933907])
        actual = estimator._compute_local_bandwidths(densities)
        np.testing.assert_array_almost_equal(expected, actual)

    def test__compute_local_bandwidths_with_non_zero_densities(self):
        estimator = MBEstimator(dimension=2, sensitivity=0.5)
        densities = np.array([1, 0, 3, 4, 5, 0])
        expected = np.array([1.66828, 1.0, 0.963182,
                             0.83414, 0.746077, 1.0])
        with warnings.catch_warnings(record=True):
            actual = estimator._compute_local_bandwidths(densities)
        np.testing.assert_array_almost_equal(expected, actual)


class ModifiedBreimanEstimatorImpAbstractTest(object):
    def setUp(self):
        super(ModifiedBreimanEstimatorImpAbstractTest, self).setUp()
        self._estimator_class = None

    def test_estimate_gaussian(self):
        xi_s = np.array([[-1, -1], [1, 1], [0, 0]])
        x_s = np.array([[0, 0], [1, 1], [0, 1]])
        local_bandwidths = np.array([10, 20, 50])
        general_bandwidth = 0.5
        kernel = Gaussian()
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=kernel,
            local_bandwidths=local_bandwidths, general_bandwidth=general_bandwidth
        )
        actual = estimator.estimate()

        expected_densities = np.array([0.00264898, 0.0024237, 0.00253281])
        expected_num_patterns_used_for_density = np.array([3, 3, 3])
        expected_local_bandwidths = np.array([10, 20, 50])

        np.testing.assert_array_almost_equal(actual.densities, expected_densities)
        np.testing.assert_array_almost_equal(
            actual.num_used_patterns,
            expected_num_patterns_used_for_density
        )
        np.testing.assert_array_almost_equal(actual.local_bandwidths, expected_local_bandwidths)


class Test_MBEEstimator_Python(ModifiedBreimanEstimatorImpAbstractTest, TestCase):
    def setUp(self):
        super(Test_MBEEstimator_Python, self).setUp()
        self._estimator_class = _MBEEstimator_Python
        _utils.set_num_threads(2)

    def tearDown(self):
        _utils.reset_num_threads()

    def test_estimate_pattern_gaussian(self):
        xi_s = np.array([[-1, -1], [1, 1], [0, 0]])
        x_s = np.array([[0, 0], [1, 1]])
        local_bandwidths = np.array([10, 20, 50])
        general_bandwidth = 0.5
        kernel = Gaussian()
        pattern = x_s[0]
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=kernel,
            local_bandwidths=local_bandwidths, general_bandwidth=general_bandwidth
        )
        actual_density, actual_num_used_patterns = estimator._estimate_pattern(pattern)

        expected_density = 0.00264898 * 3
        expected_num_used_patterns = 3

        self.assertAlmostEqual(actual_density, expected_density)
        self.assertEqual(actual_num_used_patterns, expected_num_used_patterns)


class Test_MBEEstimator_C(ModifiedBreimanEstimatorImpAbstractTest, TestCase):
    def setUp(self):
        super(Test_MBEEstimator_C, self).setUp()
        self._estimator_class = _MBEEstimator_C
        _utils.set_num_threads(2)

    def tearDown(self):
        _utils.reset_num_threads()
