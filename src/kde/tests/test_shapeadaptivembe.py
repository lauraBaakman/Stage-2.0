from unittest import TestCase, skip
import warnings

import numpy as np

import kde
from kde.kernels.shapeadaptivegaussian import ShapeAdaptiveGaussian
from kde.kernels.shapeadaptiveepanechnikov import ShapeAdaptiveEpanechnikov
from kde.kernels.testKernel import TestKernel
from kde.parzen import _ParzenEstimator_Python, _ParzenEstimator_C
from kde.sambe import \
    ShapeAdaptiveMBE, \
    _ShapeAdaptiveMBE_C, \
    _ShapeAdaptiveMBE_Python


class TestShapeAdaptiveMBE(TestCase):

    def estimate_test_helper(self, pilot_implementation, final_implementation):
        xi_s = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        x_s = xi_s
        pilot_kernel = TestKernel
        final_kernel = ShapeAdaptiveGaussian
        number_of_grid_points = 2
        sensitivity = 0.5
        estimator = ShapeAdaptiveMBE(
            pilot_kernel_class=pilot_kernel, pilot_estimator_implementation=pilot_implementation,
            kernel_class=final_kernel, final_estimator_implementation=final_implementation,
            dimension=2, number_of_grid_points=number_of_grid_points,
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

    def test_estimate_python_C(self):
        self.estimate_test_helper(_ParzenEstimator_Python, _ShapeAdaptiveMBE_C)

    def test_estimate_C_python(self):
        self.estimate_test_helper(_ParzenEstimator_C, _ShapeAdaptiveMBE_Python)

    def test_estimate_C_C(self):
        self.estimate_test_helper(_ParzenEstimator_C, _ShapeAdaptiveMBE_C)


class ShapeAdaptiveMBEImpAbstractTest(object):

    def setUp(self):
        super().setUp()
        self._estimator_class = None

    def test_estimate_epanechnikov(self):
        xi_s = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        x_s = xi_s
        local_bandwidths = np.array([0.84089642,
                                     1.18920712,
                                     1.18920712,
                                     0.84089642])
        general_bandwidth = 0.721347520444482
        kernel = ShapeAdaptiveEpanechnikov
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=kernel,
            local_bandwidths=local_bandwidths, general_bandwidth=general_bandwidth
        )
        actual = estimator.estimate()
        expected = np.array([0.567734888282212,
                             0.283867145804328,
                             0.283867145804328,
                             0.567734888282212])
        np.testing.assert_array_almost_equal(actual, expected)

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
        super().setUp()
        self._estimator_class = _ShapeAdaptiveMBE_Python

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


class Test_ShapeAdaptiveMBE_C(ShapeAdaptiveMBEImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
        self._estimator_class = _ShapeAdaptiveMBE_C

    def test_warning(self):
        try:
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
            estimator.estimate()
        except ValueError:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

