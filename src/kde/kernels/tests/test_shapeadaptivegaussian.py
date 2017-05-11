from unittest import TestCase, skip

import numpy as np

from kde.kernels.kernel import KernelException
from kde.kernels.shapeadaptivegaussian import \
    ShapeAdaptiveGaussian, \
    _ShapeAdaptiveGaussian_C, \
    _ShapeAdaptiveGaussian_Python


class TestShapeAdaptiveGaussian(TestCase):

    def test_to_C_enum(self):
        actual = ShapeAdaptiveGaussian.to_C_enum()
        expected = 4
        self.assertEqual(actual, expected)

    def test_default_implementation(self):
        H = np.array([
            [0.080225998475784, 0.000182273891304],
            [0.000182273891304, 0.081385767033078]
                ])
        x = np.array([0.05, 0.05])
        expected = 16.649507462597573
        actual = ShapeAdaptiveGaussian(H).evaluate(x)
        self.assertAlmostEqual(actual, expected)

    @skip("There is no C implementation of ShapeAdaptiveGaussian.")
    def test_alternative_implementation(self):
        H = np.array([
            [0.080225998475784, 0.000182273891304],
            [0.000182273891304, 0.081385767033078]
                ])
        x = np.array([0.05, 0.05])
        expected = 16.649507462597573
        actual = ShapeAdaptiveGaussian(H, _ShapeAdaptiveGaussian_C).evaluate(x)
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_bandwidth_matrix_0(self):
        matrix = np.random.rand(3, 3)
        actual = ShapeAdaptiveGaussian._validate_bandwidth_matrix(matrix)
        self.assertIsNone(actual)

    def test_evaluate_bandwidth_matrix_1(self):
        matrix = np.random.rand(3)
        try:
            ShapeAdaptiveGaussian._validate_bandwidth_matrix(matrix)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_evaluate_bandwidth_matrix_2(self):
        matrix = np.random.rand(3, 4, 5)
        try:
            ShapeAdaptiveGaussian._validate_bandwidth_matrix(matrix)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_evaluate_bandwidth_matrix_3(self):
        matrix = np.random.rand(3, 6)
        try:
            ShapeAdaptiveGaussian._validate_bandwidth_matrix(matrix)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_evaluate_bandwidth_matrix_4(self):
        matrix = np.random.rand(6, 3)
        try:
            ShapeAdaptiveGaussian._validate_bandwidth_matrix(matrix)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')


class Test_ShapeAdaptiveGaussian(TestCase):

    def test_dimension(self):
        matrix = np.random.rand(3, 3)
        actual = ShapeAdaptiveGaussian(matrix).dimension
        expected = 3
        self.assertEqual(actual, expected)

    def test_validate_patterns_0(self):
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        patterns = np.random.rand(10, dimension)
        actual = ShapeAdaptiveGaussian(bandwidth_matrix)._validate_patterns(patterns)
        self.assertIsNone(actual)

    def test_validate_patterns_1(self):
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        patterns = np.random.rand(1, dimension)
        actual = ShapeAdaptiveGaussian(bandwidth_matrix)._validate_patterns(patterns)
        self.assertIsNone(actual)

    def test_validate_patterns_2(self):
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        patterns = np.random.rand(10, 4)
        try:
            ShapeAdaptiveGaussian(bandwidth_matrix)._validate_patterns(patterns)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_validate_patterns_3(self):
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        patterns = np.random.rand(10, 2)
        try:
            ShapeAdaptiveGaussian(bandwidth_matrix)._validate_patterns(patterns)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_validate_patterns_4(self):
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        patterns = np.random.rand(1, 4)
        try:
            ShapeAdaptiveGaussian(bandwidth_matrix)._validate_patterns(patterns)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_validate_patterns_5(self):
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        patterns = np.random.rand(1, 2)
        try:
            ShapeAdaptiveGaussian(bandwidth_matrix)._validate_patterns(patterns)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_evaluate_0(self):
        # Single pattern, local bandwidth = 1
        H = np.array([[4, 2],
                      [7, 6]])
        x = np.array([0.05, 0.05])
        expected = 0.015914499621880
        actual = ShapeAdaptiveGaussian(H).evaluate(x)
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_1(self):
        # Multiple patterns, local bandwidth = 1
        H = np.array([[4, 2],
                      [7, 6]])
        x = np.array([[0.05, 0.05], [0.02, 0.03], [0.04, 0.05]])
        expected = np.array([
            0.015914499621880,
            0.015914340477679,
            0.015913385645896
        ])
        actual = ShapeAdaptiveGaussian(H).evaluate(x)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_2(self):
        # Single pattern, local bandwidth \neq 1
        H = np.array([[4, 2],
                      [7, 6]])
        bandwidth = 0.5
        x = np.array([0.05, 0.05])
        expected = 0.063646063731720
        actual = ShapeAdaptiveGaussian(H).evaluate(x, local_bandwidth=bandwidth)
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_3(self):
        # Multiple patterns, local bandwidth \neq 1
        H = np.array([[4, 2],
                      [7, 6]])
        local_bandwidths = np.array([0.5, 0.7, 0.2])
        x = np.array([[0.05, 0.05], [0.02, 0.03], [0.04, 0.05]])
        expected = np.array([
            0.063646063731720,
            0.032475795183358,
            0.396571536389524
        ])
        actual = ShapeAdaptiveGaussian(H).evaluate(x, local_bandwidth=local_bandwidths)
        np.testing.assert_array_almost_equal(actual, expected)


class ShapeAdaptiveGaussianImpAbstractTest(object):

    def test_evaluate_0(self):
        # Single pattern, local bandwidth = 1
        H = np.array([[4, 2],
                      [7, 6]])
        x = np.array([0.05, 0.05])
        expected = 0.015914499621880
        actual = self._kernel_class(H).evaluate(x)
        self.assertAlmostEqual(expected, actual)

    def test_evaluate_1(self):
        # Multiple patterns, local bandwidth = 1
        H = np.array([[4, 2],
                      [7, 6]])
        x = np.array([[0.05, 0.05], [0.02, 0.03], [0.04, 0.05]])
        expected = np.array([
            0.015914499621880,
            0.015914340477679,
            0.015913385645896
        ])
        actual = self._kernel_class(H).evaluate(x)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_2(self):
        # Single pattern, local bandwidth \neq 1
        H = np.array([[4, 2],
                      [7, 6]])
        bandwidth = 0.5
        x = np.array([0.05, 0.05])
        expected = 0.063646063731720
        actual = self._kernel_class(H).evaluate(x, local_bandwidth=bandwidth)
        self.assertAlmostEqual(expected, actual)

    def test_evaluate_3(self):
        # Multiple patterns, local bandwidth \neq 1
        H = np.array([[4, 2],
                      [7, 6]])
        local_bandwidths = np.array([0.5, 0.7, 0.2])
        x = np.array([[0.05, 0.05], [0.02, 0.03], [0.04, 0.05]])
        expected = np.array([
            0.063646063731720,
            0.032475795183358,
            0.396571536389524
        ])
        actual = self._kernel_class(H).evaluate(x, local_bandwidth=local_bandwidths)
        np.testing.assert_array_almost_equal(actual, expected)


class ShapeAdaptiveGaussian_Python(ShapeAdaptiveGaussianImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._kernel_class = _ShapeAdaptiveGaussian_Python


@skip("There is no C implementation of ShapeAdaptiveGaussian.")
class ShapeAdaptiveGaussian_C(ShapeAdaptiveGaussianImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._kernel_class = _ShapeAdaptiveGaussian_C
