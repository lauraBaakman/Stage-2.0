from unittest import TestCase

import numpy as np
import warnings

import kde
from kde.kernels.shapeadaptivegaussian import ShapeAdaptiveGaussian
from kde.kernels.testKernel import TestKernel
from kde.parzen import _ParzenEstimator_Python, _ParzenEstimator_C
from kde.sambe import \
    SAMBEstimator, \
    _ShapeAdaptiveMBE_C, \
    _ShapeAdaptiveMBE_Python
import kde.utils._utils as _utils

_num_threads = 2


class TestShapeAdaptiveMBE(TestCase):

    def setUp(self):
        _utils.set_num_threads(_num_threads)
        warnings.simplefilter("always")

    def tearDown(self):
        _utils.reset_num_threads()

    def estimate_test_helper(self, pilot_implementation, final_implementation):
        xi_s = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        x_s = np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [2, 2]
        ])
        pilot_kernel = TestKernel
        final_kernel = ShapeAdaptiveGaussian
        number_of_grid_points = 2
        sensitivity = 0.5
        estimator = SAMBEstimator(
            pilot_kernel_class=pilot_kernel, pilot_estimator_implementation=pilot_implementation,
            kernel_class=final_kernel, final_estimator_implementation=final_implementation,
            dimension=2,
            number_of_grid_points=number_of_grid_points,
            kernel_radius_fraction=0.2772588722239781,
            sensitivity=sensitivity,
            pilot_window_width_method=kde.utils.automaticWindowWidthMethods.ferdosi
        )
        actual = estimator.estimate(xi_s=xi_s, x_s=x_s)

        expected_densities = np.array([
            0.143018801262988,
            0.077446155260620,
            0.143018801262988,
            0.014115079016664
        ])
        expected_num_patterns_used_for_density = np.array([4, 4, 4, 3])
        expected_xis = xi_s

        np.testing.assert_array_almost_equal(actual.densities, expected_densities)
        np.testing.assert_array_almost_equal(
            actual.num_used_patterns,
            expected_num_patterns_used_for_density
        )
        np.testing.assert_array_almost_equal(actual.xis, expected_xis)

    def test_estimate_python_python(self):
        with warnings.catch_warnings(record=True):
            self.estimate_test_helper(_ParzenEstimator_Python, _ShapeAdaptiveMBE_Python)

    # Different result due to the KD tree which gives an approximation
    def test_estimate_python_C(self):
        with warnings.catch_warnings(record=True):
            xi_s = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            x_s = np.array([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [2, 2]
            ])
            pilot_kernel = TestKernel
            final_kernel = ShapeAdaptiveGaussian
            number_of_grid_points = 2
            sensitivity = 0.5
            estimator = SAMBEstimator(
                pilot_kernel_class=pilot_kernel, pilot_estimator_implementation=_ParzenEstimator_Python,
                kernel_class=final_kernel, final_estimator_implementation=_ShapeAdaptiveMBE_C,
                dimension=2,
                number_of_grid_points=number_of_grid_points, kernel_radius_fraction=0.2772588722239781,
                sensitivity=sensitivity,
                pilot_window_width_method=kde.utils.automaticWindowWidthMethods.ferdosi
            )
            actual = estimator.estimate(xi_s=xi_s, x_s=x_s)

            expected_densities = np.array([
                0.143018801266957,
                0.077446155261498,
                0.077446155261498,
                0.186693239495190,
                0.017000356330535
            ])
            expected_num_patterns_used_for_density = np.array([4, 4, 4, 4, 4])
            expected_xis = xi_s

            np.testing.assert_array_almost_equal(actual.densities, expected_densities)
            np.testing.assert_array_almost_equal(
                actual.num_used_patterns,
                expected_num_patterns_used_for_density
            )
            np.testing.assert_array_almost_equal(actual.xis, expected_xis)

    def test_estimate_C_python(self):
        with warnings.catch_warnings(record=True):
            self.estimate_test_helper(_ParzenEstimator_C, _ShapeAdaptiveMBE_Python)

    # Different result due to the KD tree which gives an approximation
    def test_estimate_C_C(self):
        xi_s = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        x_s = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 2]
        ])
        pilot_densities = np.array([1.33209861,  0.6660493,  0.6660493,  1.33209861])
        general_bandwidth = 0.72134752044448169
        pilot_kernel = TestKernel
        final_kernel = ShapeAdaptiveGaussian
        number_of_grid_points = 2
        sensitivity = 0.5
        estimator = SAMBEstimator(
            pilot_kernel_class=pilot_kernel, pilot_estimator_implementation=_ParzenEstimator_C,
            kernel_class=final_kernel, final_estimator_implementation=_ShapeAdaptiveMBE_C,
            dimension=2,
            number_of_grid_points=number_of_grid_points, kernel_radius_fraction=0.2772588722239781,
            sensitivity=sensitivity,
            pilot_window_width_method=kde.utils.automaticWindowWidthMethods.ferdosi
        )
        actual = estimator.estimate(
            xi_s=xi_s, x_s=x_s,
            pilot_densities=pilot_densities, general_bandwidth=general_bandwidth
        )
        expected_densities = np.array([
            0.143018801266957,
            0.077446155261498,
            0.077446155261498,
            0.186693239495190,
            0.017000356330535
        ])
        expected_num_patterns_used_for_density = np.array([4, 4, 4, 4, 4])
        expected_xis = xi_s

        np.testing.assert_array_almost_equal(actual.densities, expected_densities)
        np.testing.assert_array_almost_equal(
            actual.num_used_patterns,
            expected_num_patterns_used_for_density
        )
        np.testing.assert_array_almost_equal(actual.xis, expected_xis)

    def test_estimate_C_C_dont_pass_pilot_densities(self):
        with warnings.catch_warnings(record=True):
            xi_s = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            x_s = np.array([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [2, 2]
            ])
            pilot_kernel = TestKernel
            final_kernel = ShapeAdaptiveGaussian
            number_of_grid_points = 2
            sensitivity = 0.5
            estimator = SAMBEstimator(
                pilot_kernel_class=pilot_kernel, pilot_estimator_implementation=_ParzenEstimator_C,
                kernel_class=final_kernel, final_estimator_implementation=_ShapeAdaptiveMBE_C,
                dimension=2,
                number_of_grid_points=number_of_grid_points, kernel_radius_fraction=0.2772588722239781,
                sensitivity=sensitivity,
                pilot_window_width_method=kde.utils.automaticWindowWidthMethods.ferdosi
            )
            actual = estimator.estimate(
                xi_s=xi_s, x_s=x_s
            )
            expected_xis = xi_s
            expected_densities = np.array([
                0.143018801266957,
                0.077446155261498,
                0.077446155261498,
                0.186693239495190,
                0.017000356330535
            ])
            np.testing.assert_array_almost_equal(actual.densities, expected_densities)
            np.testing.assert_array_almost_equal(actual.xis, expected_xis)


class ShapeAdaptiveMBEImpAbstractTest(object):

    def setUp(self):
        super(ShapeAdaptiveMBEImpAbstractTest, self).setUp()
        _utils.set_num_threads(_num_threads)
        self._estimator_class = None

    def tearDown(self):
        _utils.reset_num_threads()


class Test_ShapeAdaptiveMBE_Python(ShapeAdaptiveMBEImpAbstractTest, TestCase):
    def setUp(self):
        super(Test_ShapeAdaptiveMBE_Python, self).setUp()
        self._estimator_class = _ShapeAdaptiveMBE_Python
        _utils.set_num_threads(_num_threads)

    def tearDown(self):
        _utils.reset_num_threads()

    def test_estimate_gaussian(self):
        xi_s = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        x_s = np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [2, 2]
        ])
        local_bandwidths = np.array([0.84089642,
                                     1.18920712,
                                     1.18920712,
                                     0.84089642])
        general_bandwidth = 0.721347520444482
        kernel = ShapeAdaptiveGaussian
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=kernel,
            local_bandwidths=local_bandwidths, general_bandwidth=general_bandwidth
        )
        actual = estimator.estimate()

        expected_densities = np.array([
            0.143018801262988,
            0.077446155260620,
            0.143018801262988,
            0.014115079016664
        ])
        expected_num_patterns_used_for_density = np.array([4, 4, 4, 3])

        np.testing.assert_array_almost_equal(actual.densities, expected_densities)
        np.testing.assert_array_almost_equal(
            actual.num_used_patterns,
            expected_num_patterns_used_for_density
        )

    def test_compute_kernel_terms(self):
        xi_s = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        x_s = np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [2, 2]
        ])
        local_bandwidths = np.array([0.84089642,
                                     1.18920712,
                                     1.18920712,
                                     0.84089642])
        general_bandwidth = 0.721347520444482
        kernel = ShapeAdaptiveGaussian
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=kernel,
            local_bandwidths=local_bandwidths, general_bandwidth=general_bandwidth
        )
        actual = estimator._compute_kernel_terms()
        expected = np.array([
            [4.32559683e-01, 6.96954147e-02, 6.96954147e-02, 1.24451697e-04],
            [4.49182023e-02, 2.16279842e-01, 3.66854145e-03, 4.49182023e-02],
            [1.24451697e-04, 6.96954147e-02, 6.96954147e-02, 4.32559683e-01],
            [2.96391110e-15, 2.81679173e-02, 2.81679173e-02, 1.24451697e-04]
        ])
        np.testing.assert_array_almost_equal(actual, expected)


class Test_ShapeAdaptiveMBE_C(ShapeAdaptiveMBEImpAbstractTest, TestCase):
    def setUp(self):
        super(Test_ShapeAdaptiveMBE_C, self).setUp()
        self._estimator_class = _ShapeAdaptiveMBE_C
        _utils.set_num_threads(_num_threads)

    def tearDown(self):
        _utils.reset_num_threads()

    """
        Different test than the Python implementation, since the C implementation uses a KD tree,
        which gives an approximation.
    """
    def test_estimate_gaussian(self):
        xi_s = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        x_s = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 2]
        ])
        local_bandwidths = np.array([0.84089642,
                                     1.18920712,
                                     1.18920712,
                                     0.84089642])
        general_bandwidth = 0.721347520444482
        kernel = ShapeAdaptiveGaussian
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=kernel,
            local_bandwidths=local_bandwidths, general_bandwidth=general_bandwidth
        )
        actual = estimator.estimate()
        expected_densities = np.array([
            0.143018801266957,
            0.077446155261498,
            0.077446155261498,
            0.186693239495190,
            0.017000356330535
        ])
        expected_num_patterns_used_for_density = np.array([4, 4, 4, 4, 4])

        expected_eigen_vectors = np.array([
            [[+0.707106781186547, +0.707106781186547], [-0.707106781186547, +0.707106781186547]],
            [[+0.707106781186547, -0.707106781186547], [+0.707106781186547, +0.707106781186547]],
            [[+0.707106781186547, +0.707106781186547], [-0.707106781186547, +0.707106781186547]],
            [[+0.707106781186547, +0.707106781186547], [-0.707106781186547, +0.707106781186547]],
        ])
        expected_eigen_values = np.array([
            [0.350208287700000, 1.050624863100000],
            [0.495269309400000, 1.485807928200000],
            [0.495269309400000, 1.485807928200000],
            [0.350208287700000, 1.050624863100000],
        ])

        np.testing.assert_array_almost_equal(actual.densities, expected_densities)
        np.testing.assert_array_almost_equal(
            actual.num_used_patterns,
            expected_num_patterns_used_for_density
        )
        np.testing.assert_array_almost_equal(actual.densities, expected_densities)
        np.testing.assert_array_almost_equal(
            np.sort(actual.eigen_values, axis=1),
            np.sort(expected_eigen_values, axis=1)
        )
        np.testing.assert_array_almost_equal(actual.eigen_vectors, expected_eigen_vectors)

    def test__reshape_eigen_vectors(self):
        eigen_vectors = np.array([
            [2, 3, 4, 5],
            [6, 7, 7, 8],
            [9, 10, 11, 12],
        ])
        xi_s = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
        ])
        x_s = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 2]
        ])
        local_bandwidths = np.array([0.84089642,
                                     1.18920712,
                                     1.18920712,
                                     0.84089642])
        general_bandwidth = 0.721347520444482
        kernel = ShapeAdaptiveGaussian
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=kernel,
            local_bandwidths=local_bandwidths, general_bandwidth=general_bandwidth
        )
        actual = estimator._reshape_eigen_vectors(eigen_vectors)
        expected = np.array([
            [[2, 3], [4, 5]],
            [[6, 7], [7, 8]],
            [[9, 10], [11, 12]],
        ])
        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_array_equal(actual, expected)
