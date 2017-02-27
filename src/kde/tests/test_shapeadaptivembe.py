from unittest import TestCase, skip

import numpy as np

import kde
from kde.kernels.testKernel import TestKernel
from kde.kernels.epanechnikov import Epanechnikov
from kde.kernels.gaussian import Gaussian
from kde.parzen import _ParzenEstimator_Python, _ParzenEstimator_C
from kde.shapeadaptivembe import \
    ShapeAdaptiveMBE, \
    _ShapeAdaptiveMBE, \
    _ShapeAdaptiveMBE_C, \
    _ShapeAdaptiveMBE_Python


class TestShapeAdaptiveMBE(TestCase):

    def estimate_test_helper(self, pilot_implementation, final_implementation):
        xi_s = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        x_s = np.array([[0, 0], [1, 1]])
        pilot_kernel = TestKernel
        final_kernel = Gaussian
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
        expected = np.array([0.511743799443552, 0.511743799443552])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_estimate_python_python(self):
        self.estimate_test_helper(_ParzenEstimator_Python, _ShapeAdaptiveMBE_Python)

    @skip("The C implementation of SAMBE has not yet been written.")
    def test_estimate_python_C(self):
        self.estimate_test_helper(_ParzenEstimator_Python, _ShapeAdaptiveMBE_C)

    @skip("The C implementation of SAMBE has not yet been written.")
    def test_estimate_C_python(self):
        self.estimate_test_helper(_ParzenEstimator_C, _ShapeAdaptiveMBE_Python)

    @skip("The C implementation of SAMBE has not yet been written.")
    def test_estimate_C_C(self):
        self.estimate_test_helper(_ParzenEstimator_C, _ShapeAdaptiveMBE_C)

class ShapeAdaptiveMBEImpAbstractTest(object):

    def setUp(self):
        super().setUp()
        self._estimator_class = None

    @skip("The Epanechnikov kernel is not yet shape adaptive.")
    def test_estimate_epanechnikov(self):
        self.fail("Test not yet implemented.")

    def test_estimate_gaussian(self):
        xi_s = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        x_s = np.array([[0, 0], [1, 1]])
        local_bandwidths = np.array([10, 20, 50])
        general_bandwidth = 0.5
        kernel = Gaussian
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=kernel,
            local_bandwidths=local_bandwidths, general_bandwidth=general_bandwidth
        )
        actual = estimator.estimate()
        expected = np.array([0.511743799443552, 0.511743799443552])
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
        kernel = Gaussian
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=kernel,
            local_bandwidths=local_bandwidths, general_bandwidth=h
        )
        actual = estimator._determine_kernel_shape(pattern)
        expected = np.array([
            [0.263583071129392,  -0.131791535564696],
            [-0.131791535564696,   0.263583071129392]
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
        kernel = Gaussian
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=kernel,
            local_bandwidths=local_bandwidths, general_bandwidth=h
        )
        actual = estimator._estimate_pattern(pattern=pattern)
        expected = 0.511743799443552
        self.assertAlmostEqual(actual, expected)


@skip("The C implementation of SAMBE has not yet been written.")
class Test_ShapeAdaptiveMBE_C(ShapeAdaptiveMBEImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
        self._estimator_class = _ShapeAdaptiveMBE_C
