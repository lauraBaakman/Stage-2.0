from unittest import TestCase, skip

import numpy as np

import kde
from kde.kernels.testKernel import TestKernel
from kde.kernels.epanechnikov import Epanechnikov
from kde.kernels.gaussian import Gaussian
from kde.parzen import _ParzenEstimator_Python, _ParzenEstimator_C
from kde.shapeadaptivembe import \
    ShapeAdaptiveModifiedBreimanEstimator, \
    _ShapeAdaptiveModifiedBreimanEstimator, \
    _ShapeAdaptiveModifiedBreimanEstimator_C, \
    _ShapeAdaptiveModifiedBreimanEstimator_Python


class TestShapeAdaptiveMBE(TestCase):

    def estimate_test_helper(self, pilot_implementation, final_implementation):
        self.fail("Test not yet implemented.")
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
            pilot_window_width_method=kde.utils.automaticWindowWidthMethods.ferdosi
        )
        actual = estimator.estimate(xi_s=xi_s, x_s=x_s)
        expected = None
        np.testing.assert_array_almost_equal(actual, expected)

    def test_estimate_python_python(self):
        self.estimate_test_helper(_ParzenEstimator_Python, _ShapeAdaptiveModifiedBreimanEstimator_Python)

    @skip("The C implementation of SAMBE has not yet been written.")
    def test_estimate_python_C(self):
        self.estimate_test_helper(_ParzenEstimator_Python, _ShapeAdaptiveModifiedBreimanEstimator_C)

    @skip("The C implementation of SAMBE has not yet been written.")
    def test_estimate_C_python(self):
        self.estimate_test_helper(_ParzenEstimator_C, _ShapeAdaptiveModifiedBreimanEstimator_Python)

    @skip("The C implementation of SAMBE has not yet been written.")
    def test_estimate_C_C(self):
        self.estimate_test_helper(_ParzenEstimator_C, _ShapeAdaptiveModifiedBreimanEstimator_C)

class ShapeAdaptiveMBEImpAbstractTest(object):

    def setUp(self):
        super().setUp()
        self._estimator_class = None

    @skip("The Epanechnikov kernel is not yet shape adaptive.")
    def test_estimate_epanechnikov(self):
        self.fail("Test not yet implemented.")

    def test_estimate_gaussian(self):
        self.fail("Test not yet implemented.")
        xi_s = np.array([[-1, -1], [1, 1], [0, 0]])
        x_s = np.array([[0, 0], [1, 1], [0, 1]])
        local_bandwidths = np.array([10, 20, 50])
        general_bandwidth = 0.5
        kernel = Gaussian
        estimator = self._estimator_class(
            xi_s=xi_s, x_s=x_s, dimension=2,
            kernel=kernel,
            local_bandwidths=local_bandwidths, general_bandwidth=general_bandwidth
        )
        actual = estimator.estimate()
        expected = None
        np.testing.assert_array_almost_equal(actual, expected)


class Test_ShapeAdaptiveMBE_Python(ShapeAdaptiveMBEImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
        self._estimator_class = _ShapeAdaptiveModifiedBreimanEstimator_Python

    def test__determine_kernel_shape(self):
        self.fail("Test not yet implemented.")

    def test__estimate_pattern(self):
        self.fail("Test not yet implemented.")


@skip("The C implementation of SAMBE has not yet been written.")
class Test_ShapeAdaptiveMBE_C(ShapeAdaptiveMBEImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
        self._estimator_class = _ShapeAdaptiveModifiedBreimanEstimator_C
