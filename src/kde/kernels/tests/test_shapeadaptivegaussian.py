from unittest import TestCase, skip

import numpy as np

from kde.kernels.kernel import KernelException
from kde.kernels.shapeadaptivegaussian import \
    ShapeAdaptiveGaussian, \
    _ShapeAdaptiveGaussian, \
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

    def test_define_and_validate_patterns_0(self):
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        patterns = np.random.rand(10, dimension)
        expected = patterns
        actual = ShapeAdaptiveGaussian(bandwidth_matrix)._define_and_validate_patterns(patterns)
        np.testing.assert_array_equal(actual, expected)

    def test_define_and_validate_patterns_1(self):
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        patterns = np.random.rand(1, dimension)
        expected = patterns
        actual = ShapeAdaptiveGaussian(bandwidth_matrix)._define_and_validate_patterns(patterns)
        np.testing.assert_array_equal(actual, expected)

    def test_define_and_validate_patterns_2(self):
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        patterns = np.random.rand(10, 4)
        try:
            ShapeAdaptiveGaussian(bandwidth_matrix)._define_and_validate_patterns(patterns)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_define_and_validate_patterns_3(self):
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        patterns = np.random.rand(10, 2)
        try:
            ShapeAdaptiveGaussian(bandwidth_matrix)._define_and_validate_patterns(patterns)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_define_and_validate_patterns_4(self):
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        patterns = np.random.rand(1, 4)
        try:
            ShapeAdaptiveGaussian(bandwidth_matrix)._define_and_validate_patterns(patterns)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_define_and_validate_patterns_5(self):
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        patterns = np.random.rand(1, 2)
        try:
            ShapeAdaptiveGaussian(bandwidth_matrix)._define_and_validate_patterns(patterns)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_define_and_validate_patterns_6(self):
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        patterns = np.array([1.0, 2.0, 3.0])
        actual = ShapeAdaptiveGaussian(bandwidth_matrix)._define_and_validate_patterns(patterns)
        expected = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(expected, actual)

    def test_define_and_validate_patterns_7(self):
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        patterns = np.array([[1.0, 2.0, 3.0]])
        actual = ShapeAdaptiveGaussian(bandwidth_matrix)._define_and_validate_patterns(patterns)
        expected = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(expected, actual)

    def test_define_and_validate_local_bandwidths_0(self):
        # Local bandwidth is none, one pattern
        input_local_bandwidth = None
        dimension = 3
        patterns = np.random.rand(1, dimension)
        bandwidth_matrix = np.random.rand(dimension, dimension)

        expected = np.array([1.0])

        actual = ShapeAdaptiveGaussian(bandwidth_matrix)._define_and_validate_local_bandwidths(
            input_local_bandwidth, patterns
        )

        np.testing.assert_array_equal(actual, expected)

    def test_define_and_validate_local_bandwidths_1(self):
        # Local bandwidth is none, multiple patterns
        input_local_bandwidth = None
        dimension = 7
        patterns = np.random.rand(5, dimension)
        bandwidth_matrix = np.random.rand(dimension, dimension)

        expected_local_bandwidths = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        actual_local_bandwidths = ShapeAdaptiveGaussian(bandwidth_matrix)._define_and_validate_local_bandwidths(
            input_local_bandwidth, patterns
        )

        np.testing.assert_array_equal(actual_local_bandwidths, expected_local_bandwidths)

    def test_define_and_validate_local_bandwidths_2(self):
        # Local bandwidth is a double, one pattern
        input_local_bandwidth = 0.5
        dimension = 3
        patterns = np.random.rand(1, dimension)
        bandwidth_matrix = np.random.rand(dimension, dimension)

        expected = np.array([0.5])

        actual = ShapeAdaptiveGaussian(bandwidth_matrix)._define_and_validate_local_bandwidths(
            input_local_bandwidth, patterns
        )

        np.testing.assert_array_equal(actual, expected)

    def test_define_and_validate_local_bandwidths_3(self):
        # Local bandwidth is a double, multiple patterns
        input_local_bandwidth = 0.5
        dimension = 4
        patterns = np.random.rand(5, dimension)
        bandwidth_matrix = np.random.rand(dimension, dimension)

        try:
            ShapeAdaptiveGaussian(bandwidth_matrix)._define_and_validate_local_bandwidths(
                input_local_bandwidth, patterns
            )
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_define_and_validate_local_bandwidths_4(self):
        # Local bandwidth is a 1x1 array, single pattern
        input_local_bandwidth = np.array([0.5])
        dimension = 2
        patterns = np.random.rand(1, dimension)
        bandwidth_matrix = np.random.rand(dimension, dimension)

        expected_local_bandwidths = np.array([0.5])

        actual_local_bandwidths = ShapeAdaptiveGaussian(bandwidth_matrix)._define_and_validate_local_bandwidths(
            input_local_bandwidth, patterns
        )

        np.testing.assert_array_equal(actual_local_bandwidths, expected_local_bandwidths)

    def test_define_and_validate_local_bandwidths_5(self):
        # Local bandwidth is a too small array
        input_local_bandwidth = np.random.rand(3)
        dimension = 5
        patterns = np.random.rand(5, dimension)
        bandwidth_matrix = np.random.rand(dimension, dimension)

        try:
            ShapeAdaptiveGaussian(bandwidth_matrix)._define_and_validate_local_bandwidths(
                input_local_bandwidth, patterns
            )
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_define_and_validate_local_bandwidths_6(self):
        # Local bandwidth is an array of the correct size
        input_local_bandwidths = np.random.rand(5)
        dimension = 6
        patterns = np.random.rand(5, dimension)
        bandwidth_matrix = np.random.rand(dimension, dimension)

        actual = ShapeAdaptiveGaussian(bandwidth_matrix)._define_and_validate_local_bandwidths(
                input_local_bandwidths, patterns
            )
        expected_local_bandwidths = input_local_bandwidths
        np.testing.assert_array_equal(actual, expected_local_bandwidths)

    def test_define_and_validate_local_bandwidths_7(self):
        # Local bandwidth is a too large array
        input_local_bandwidth = np.random.rand(10)
        dimension = 3
        patterns = np.random.rand(5, dimension)
        bandwidth_matrix = np.random.rand(dimension, dimension)

        try:
            ShapeAdaptiveGaussian(bandwidth_matrix)._define_and_validate_local_bandwidths(
                input_local_bandwidth, patterns
            )
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_define_and_validate_local_bandwidths_8(self):
        #Local bandwidth has too many dimensions
        input_local_bandwidth = np.random.rand(5, 2)
        dimension = 3
        patterns = np.random.rand(5, dimension)
        bandwidth_matrix = np.random.rand(dimension, dimension)

        try:
            ShapeAdaptiveGaussian(bandwidth_matrix)._define_and_validate_local_bandwidths(
                input_local_bandwidth, patterns
            )
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_define_and_validate_local_bandwidths_9(self):
        #Local bandwidth has too many dimensions
        input_local_bandwidth = np.random.rand(5,1)
        dimension = 3
        patterns = np.random.rand(5, dimension)
        bandwidth_matrix = np.random.rand(dimension, dimension)

        try:
            ShapeAdaptiveGaussian(bandwidth_matrix)._define_and_validate_local_bandwidths(
                input_local_bandwidth, patterns
            )
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_create_default_local_bandwidths_array_0(self):
        num_patterns = 1
        expected = np.array([1.0])

        actual = _ShapeAdaptiveGaussian._create_default_local_bandwidths_array(None, num_patterns)

        np.testing.assert_array_equal(actual, expected)

    def test_create_default_local_bandwidths_array_1(self):
        num_patterns = 5
        expected = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        actual = _ShapeAdaptiveGaussian._create_default_local_bandwidths_array(None, num_patterns)

        np.testing.assert_array_equal(actual, expected)

    def test_validate_local_bandwidths_0(self):
        # Too short
        local_bandwidths = np.random.rand(7)
        num_patterns = 10
        try:
            _ShapeAdaptiveGaussian._validate_local_bandwidths(None, local_bandwidths, num_patterns)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_validate_local_bandwidths_1(self):
        # Too long
        local_bandwidths = np.random.rand(15)
        num_patterns = 10
        try:
            _ShapeAdaptiveGaussian._validate_local_bandwidths(None, local_bandwidths, num_patterns)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_validate_local_bandwidths_3(self):
        # Fine
        num_patterns = 10
        local_bandwidths = np.random.rand(num_patterns)
        actual = _ShapeAdaptiveGaussian._validate_local_bandwidths(None, local_bandwidths, num_patterns)
        self.assertIsNone(actual)


    def test_validate_local_bandwidths_4(self):
        # 1D but too many dimensions
        num_patterns = 10
        local_bandwidths = np.random.rand(num_patterns, 1)
        try:
            _ShapeAdaptiveGaussian._validate_local_bandwidths(None, local_bandwidths, num_patterns)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_validate_local_bandwidths_5(self):
        # Too many dimensions
        num_patterns = 10
        local_bandwidths = np.random.rand(num_patterns, 3)
        try:
            _ShapeAdaptiveGaussian._validate_local_bandwidths(None, local_bandwidths, num_patterns)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_define_and_validate_input(self):
        dimension = 3
        num_patterns = 10
        patterns = np.random.rand(num_patterns, dimension)
        local_bandwidths = np.random.rand(num_patterns)
        bandwidth_matrix = np.random.rand(dimension, dimension)
        (actual_patterns, actual_bandwidths) = _ShapeAdaptiveGaussian(bandwidth_matrix)._define_and_validate_input(
            patterns, local_bandwidths
        )
        expected_patterns = patterns
        expected_bandwidths = local_bandwidths

        np.testing.assert_array_equal(actual_patterns, expected_patterns)
        np.testing.assert_array_equal(actual_bandwidths, expected_bandwidths)

    def test_evaluate_pattern_0(self):
        H = np.array([[4, 2],
                      [7, 6]])
        x = np.array([0.05, 0.05])
        local_bandwidth = 1.0
        expected = 0.015914499621880
        actual = ShapeAdaptiveGaussian(H)._evaluate_pattern(x, local_bandwidth)
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_pattern_1(self):
        H = np.array([[4, 2],
                      [7, 6]])
        local_bandwidth = 0.5
        x = np.array([0.05, 0.05])
        expected = 0.063646063731720
        actual = ShapeAdaptiveGaussian(H)._evaluate_pattern(x, local_bandwidth)
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_0(self):
        # Single pattern, local bandwidth = 1
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        x = np.array([0.05, 0.05, 0.05])
        expected = 0.015705646827791
        actual = ShapeAdaptiveGaussian(H).evaluate(x)
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_1(self):
        # Multiple patterns, local bandwidth = 1
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        x = np.array([
            [0.05, 0.05, 0.05],
            [0.02, 0.03, 0.04],
            [0.04, 0.05, 0.03]
        ])
        expected = np.array([
            0.015705646827791,
            0.015812413849947,
            0.015759235403527
        ])
        actual = ShapeAdaptiveGaussian(H).evaluate(x)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_2(self):
        # Single patterns, local bandwidth \neq 1
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        bandwidth = 0.5
        x = np.array([0.05, 0.05, 0.05])
        actual = ShapeAdaptiveGaussian(H).evaluate(x, local_bandwidths=bandwidth)
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_3(self):
        # Multiple patterns, local bandwidth \neq 1
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        local_bandwidths = np.array([0.5, 0.7, 0.2])
        x = np.array([
            [0.05, 0.05, 0.05],
            [0.02, 0.03, 0.04],
            [0.04, 0.05, 0.03]
        ])
        expected = np.array([
            0.121703390601269,
            0.125046649030584,
            0.123372947325345
        ])
        actual = ShapeAdaptiveGaussian(H).evaluate(x, local_bandwidths=local_bandwidths)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_4(self):
        # Single pattern, local bandwidth \neq 1
        H = np.array([[4, 2],
                      [7, 6]])
        bandwidth = np.array([0.5])
        x = np.array([0.05, 0.05])
        expected = 0.063646063731720
        actual = ShapeAdaptiveGaussian(H).evaluate(x, local_bandwidths=bandwidth)
        self.assertAlmostEqual(actual, expected)

class ShapeAdaptiveGaussianImpAbstractTest(object):

    def test_evaluate_0(self):
        # Single pattern, local bandwidth = 1
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        x = np.array([0.05, 0.05, 0.05])
        expected = 0.015705646827791
        actual = self._kernel_class(H).evaluate(x)
        self.assertAlmostEqual(expected, actual)

    def test_evaluate_1(self):
        # Multiple patterns, local bandwidth = 1
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        x = np.array([
            [0.05, 0.05, 0.05],
            [0.02, 0.03, 0.04],
            [0.04, 0.05, 0.03]
        ])
        expected = np.array([
            0.015705646827791,
            0.015812413849947,
            0.015759235403527
        ])
        actual = self._kernel_class(H).evaluate(x)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_2(self):
        # Single pattern, local bandwidth \neq 1
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        bandwidth = 0.5
        x = np.array([0.05, 0.05, 0.05])
        expected = 0.121703390601269
        actual = self._kernel_class(H).evaluate(x, local_bandwidths=bandwidth)
        self.assertAlmostEqual(expected, actual)

    def test_evaluate_3(self):
        # Multiple patterns, local bandwidth \neq 1
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        local_bandwidths = np.array([0.5, 0.7, 0.2])
        x = np.array([
            [0.05, 0.05, 0.05],
            [0.02, 0.03, 0.04],
            [0.04, 0.05, 0.03]
        ])
        expected = np.array([
            0.121703390601269,
            0.125046649030584,
            0.123372947325345
        ])
        actual = self._kernel_class(H).evaluate(x, local_bandwidths=local_bandwidths)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_4(self):
        # Single pattern, local bandwidth \neq 1
        H = np.array([[4, 2],
                      [7, 6]])
        bandwidth = np.array([0.5])
        x = np.array([0.05, 0.05])
        expected = 0.063646063731720
        actual = ShapeAdaptiveGaussian(H).evaluate(x, local_bandwidths=bandwidth)
        self.assertAlmostEqual(actual, expected)

class ShapeAdaptiveGaussian_Python(ShapeAdaptiveGaussianImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._kernel_class = _ShapeAdaptiveGaussian_Python

    def test__handle_return_0(self):
        densities = np.array([0.5])
        expected = 0.5
        actual = _ShapeAdaptiveGaussian_Python._handle_return(None, densities)
        self.assertEqual(actual, expected)

    def test__handle_return_1(self):
        densities = np.array([0.5, 0.2])
        expected = np.array([0.5, 0.2])
        actual = _ShapeAdaptiveGaussian_Python._handle_return(None, densities)
        np.testing.assert_array_equal(actual, expected)

    def test__compute_local_scaling_factor(self):
        H = np.array([[4, 2],
                      [7, 6]])
        local_bandwidth = 0.5
        expected = 1 / 2.5
        actual = _ShapeAdaptiveGaussian_Python(H)._compute_local_scaling_factor(local_bandwidth)
        self.assertAlmostEqual(actual, expected)

    def test__compute_local_inverse(self):
        H = np.array([[4, 2],
                      [7, 6]])
        local_bandwidth = 0.5
        expected = np.array([[1.2, -0.4],
                      [- 1.4, 0.8]])
        actual = _ShapeAdaptiveGaussian_Python(H)._compute_local_inverse(local_bandwidth)
        np.testing.assert_array_almost_equal(expected, actual)

class ShapeAdaptiveGaussian_C(ShapeAdaptiveGaussianImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._kernel_class = _ShapeAdaptiveGaussian_C
