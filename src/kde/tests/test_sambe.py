from unittest import TestCase

import numpy as np

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

    def tearDown(self):
        _utils.reset_num_threads()

    def estimate_test_helper(self, pilot_implementation, final_implementation):
        xi_s = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        x_s = xi_s
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
        expected = np.array([0.143018801263046,
                             0.077446155260620,
                             0.077446155260620,
                             0.143018801263046])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_estimate_python_python(self):
        self.estimate_test_helper(_ParzenEstimator_Python, _ShapeAdaptiveMBE_Python)

    # Different result due to the KD tree which gives an approximation
    def test_estimate_python_C(self):
        xi_s = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        x_s = xi_s
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
        expected = np.array([0.186693239491116,
                             0.077446155260620,
                             0.077446155260620,
                             0.143018801263046])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_estimate_C_python(self):
        self.estimate_test_helper(_ParzenEstimator_C, _ShapeAdaptiveMBE_Python)

    # Different result due to the KD tree which gives an approximation
    def test_estimate_C_C(self):
        xi_s = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        x_s = xi_s
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
        actual = estimator.estimate(xi_s=xi_s, x_s=x_s)
        expected = np.array([0.186693239491116,
                             0.077446155260620,
                             0.077446155260620,
                             0.143018801263046])
        np.testing.assert_array_almost_equal(actual, expected)


class ShapeAdaptiveMBEImpAbstractTest(object):

    def setUp(self):
        super(ShapeAdaptiveMBEImpAbstractTest, self).setUp()
        _utils.set_num_threads(_num_threads)
        self._estimator_class = None

    def tearDown(self):
        _utils.reset_num_threads()

    def test_estimate_gaussian(self):
        xi_s = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        x_s = xi_s
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
        expected = np.array([0.143018801263046,
                             0.077446155260620,
                             0.077446155260620,
                             0.143018801263046])
        np.testing.assert_array_almost_equal(actual, expected)


class Test_ShapeAdaptiveMBE_Python(ShapeAdaptiveMBEImpAbstractTest, TestCase):
    def setUp(self):
        super(Test_ShapeAdaptiveMBE_Python, self).setUp()
        self._estimator_class = _ShapeAdaptiveMBE_Python
        _utils.set_num_threads(_num_threads)

    def tearDown(self):
        _utils.reset_num_threads()

    def test__determine_kernel_shape(self):
        xi_s = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        x_s = np.array([[0, 0], [1, 1]])
        pattern = x_s[0]
        local_bandwidths = np.array([0.840896194313949,
                                     1.189207427458816,
                                     1.189207427458816,
                                     0.840896194313949])
        h = 0.721347520444482
        kernel = ShapeAdaptiveGaussian

        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=kernel,
            local_bandwidths=local_bandwidths, general_bandwidth=h
        )

        actual = estimator._determine_kernel_shape(pattern)
        expected = np.array([
            [0.832940370215782, -0.416470185107891],
            [- 0.416470185107891, 0.832940370215782]
        ])
        np.testing.assert_array_almost_equal(actual, expected)

    def test__estimate_pattern(self):
        xi_s = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        x_s = np.array([[0, 0], [1, 1]])
        pattern = x_s[0]
        local_bandwidths = np.array([0.840896194313949,
                                     1.189207427458816,
                                     1.189207427458816,
                                     0.840896194313949])
        h = 0.721347520444482
        kernel = ShapeAdaptiveGaussian
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=kernel,
            local_bandwidths=local_bandwidths, general_bandwidth=h
        )
        actual = estimator._estimate_pattern(pattern=pattern)
        expected = 0.143018801263046
        self.assertAlmostEqual(actual, expected)

    def test_estimate_gaussian_2(self):
        # xs != xis
        xi_s = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        x_s = np.array([[0, 0], [1, 1]])
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
        expected = np.array([0.143018801263046, 0.143018801263046])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_xis_is_not_xs(self):
        xi_s = xi_s = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        x_s = np.array([
            [0.5, 0.3],
            [0.2, 0.7],
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
        expected = np.array([0.1271447313485623, 0.1507791646676249])
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
        xi_s = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        x_s = xi_s
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
        expected = np.array([0.186693239491116,
                             0.077446155260620,
                             0.077446155260620,
                             0.143018801263046])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_xis_is_not_xs(self):
        xi_s = xi_s = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        x_s = np.array([
            [0.5, 0.3],
            [0.2, 0.7],
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
        expected = np.array([0.19859744879119276, 0.15077916466762492])
        np.testing.assert_array_almost_equal(actual, expected)
