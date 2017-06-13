from unittest import TestCase

import numpy as np

from kde.kernels.kernel import ShapeAdaptiveKernel, KernelException, ShapeAdaptiveKernel_Python, ShapeAdaptive

class TestShapeAdaptiveKernel(TestCase):
    def test_dimension(self):
        matrix = np.random.rand(3, 3)
        actual = ShapeAdaptiveKernel(matrix).dimension
        expected = 3
        self.assertEqual(actual, expected)

    def test_define_and_validate_patterns_0(self):
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        patterns = np.random.rand(10, dimension)
        expected = patterns
        actual = ShapeAdaptiveKernel(bandwidth_matrix)._define_and_validate_patterns(patterns)
        np.testing.assert_array_equal(actual, expected)

    def test_define_and_validate_patterns_1(self):
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        patterns = np.random.rand(1, dimension)
        expected = patterns
        actual = ShapeAdaptiveKernel(bandwidth_matrix)._define_and_validate_patterns(patterns)
        np.testing.assert_array_equal(actual, expected)

    def test_define_and_validate_patterns_2(self):
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        patterns = np.random.rand(10, 4)
        try:
            ShapeAdaptiveKernel(bandwidth_matrix)._define_and_validate_patterns(patterns)
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
            ShapeAdaptiveKernel(bandwidth_matrix)._define_and_validate_patterns(patterns)
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
            ShapeAdaptiveKernel(bandwidth_matrix)._define_and_validate_patterns(patterns)
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
            ShapeAdaptiveKernel(bandwidth_matrix)._define_and_validate_patterns(patterns)
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
        actual = ShapeAdaptiveKernel(bandwidth_matrix)._define_and_validate_patterns(patterns)
        expected = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(expected, actual)

    def test_define_and_validate_patterns_7(self):
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        patterns = np.array([[1.0, 2.0, 3.0]])
        actual = ShapeAdaptiveKernel(bandwidth_matrix)._define_and_validate_patterns(patterns)
        expected = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(expected, actual)

    def test_define_and_validate_patterns_8(self):
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        patterns = np.random.rand(10, dimension, 3)
        try:
            ShapeAdaptiveKernel(bandwidth_matrix)._define_and_validate_patterns(patterns)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')


    def test_define_and_validate_local_bandwidths_0(self):
        # Local bandwidth is none, one pattern
        input_local_bandwidth = None
        dimension = 3
        patterns = np.random.rand(1, dimension)
        bandwidth_matrix = np.random.rand(dimension, dimension)

        expected = np.array([1.0])

        actual = ShapeAdaptiveKernel(bandwidth_matrix)._define_and_validate_local_bandwidths(
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

        actual_local_bandwidths = ShapeAdaptiveKernel(bandwidth_matrix)._define_and_validate_local_bandwidths(
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

        actual = ShapeAdaptiveKernel(bandwidth_matrix)._define_and_validate_local_bandwidths(
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
            ShapeAdaptiveKernel(bandwidth_matrix)._define_and_validate_local_bandwidths(
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

        actual_local_bandwidths = ShapeAdaptiveKernel(bandwidth_matrix)._define_and_validate_local_bandwidths(
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
            ShapeAdaptiveKernel(bandwidth_matrix)._define_and_validate_local_bandwidths(
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

        actual = ShapeAdaptiveKernel(bandwidth_matrix)._define_and_validate_local_bandwidths(
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
            ShapeAdaptiveKernel(bandwidth_matrix)._define_and_validate_local_bandwidths(
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
            ShapeAdaptiveKernel(bandwidth_matrix)._define_and_validate_local_bandwidths(
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
            ShapeAdaptiveKernel(bandwidth_matrix)._define_and_validate_local_bandwidths(
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

        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)

        actual = ShapeAdaptiveKernel(bandwidth_matrix)._create_default_local_bandwidths_array(num_patterns)

        np.testing.assert_array_equal(actual, expected)

    def test_create_default_local_bandwidths_array_1(self):
        num_patterns = 5
        expected = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)

        actual = ShapeAdaptiveKernel(bandwidth_matrix)._create_default_local_bandwidths_array(num_patterns)

        np.testing.assert_array_equal(actual, expected)

    def test_validate_local_bandwidths_0(self):
        # Too short
        local_bandwidths = np.random.rand(7)
        num_patterns = 10
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        try:
            ShapeAdaptiveKernel(bandwidth_matrix)._validate_local_bandwidths(local_bandwidths, num_patterns)
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
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        try:
            ShapeAdaptiveKernel(bandwidth_matrix)._validate_local_bandwidths(local_bandwidths, num_patterns)
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
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        actual = ShapeAdaptiveKernel(bandwidth_matrix)._validate_local_bandwidths(local_bandwidths, num_patterns)
        self.assertIsNone(actual)


    def test_validate_local_bandwidths_4(self):
        # 1D but too many dimensions
        num_patterns = 10
        local_bandwidths = np.random.rand(num_patterns, 1)
        try:
            dimension = 3
            bandwidth_matrix = np.random.rand(dimension, dimension)
            ShapeAdaptiveKernel(bandwidth_matrix)._validate_local_bandwidths(local_bandwidths, num_patterns)
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
            dimension = 3
            bandwidth_matrix = np.random.rand(dimension, dimension)
            ShapeAdaptiveKernel(bandwidth_matrix)._validate_local_bandwidths(local_bandwidths, num_patterns)
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
        (actual_patterns, actual_bandwidths) = ShapeAdaptiveKernel(bandwidth_matrix)._define_and_validate_input(
            patterns, local_bandwidths
        )
        expected_patterns = patterns
        expected_bandwidths = local_bandwidths

        np.testing.assert_array_equal(actual_patterns, expected_patterns)
        np.testing.assert_array_equal(actual_bandwidths, expected_bandwidths)


class TestShapeAdaptiveKernel_Python(TestCase):

    def test__handle_return_0(self):
        densities = np.array([0.5])
        expected = 0.5
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        actual = ShapeAdaptiveKernel_Python(bandwidth_matrix)._handle_return(densities)
        self.assertEqual(actual, expected)

    def test__handle_return_1(self):
        densities = np.array([0.5, 0.2])
        expected = np.array([0.5, 0.2])
        dimension = 3
        bandwidth_matrix = np.random.rand(dimension, dimension)
        actual = ShapeAdaptiveKernel_Python(bandwidth_matrix)._handle_return(densities)
        np.testing.assert_array_equal(actual, expected)

    def test__compute_local_scaling_factor(self):
        H = np.array([[4, 2],
                      [7, 6]])
        local_bandwidth = 0.5
        expected = 1 / 2.5
        actual = ShapeAdaptiveKernel_Python(H)._compute_local_scaling_factor(local_bandwidth)
        self.assertAlmostEqual(actual, expected)

    def test__compute_local_inverse(self):
        H = np.array([[4, 2],
                      [7, 6]])
        local_bandwidth = 0.5
        expected = np.array([[1.2, -0.4],
                      [- 1.4, 0.8]])
        actual = ShapeAdaptiveKernel_Python(H)._compute_local_inverse(local_bandwidth)
        np.testing.assert_array_almost_equal(expected, actual)

class TestShapeAdaptive(TestCase):
    def test_evaluate_bandwidth_matrix_0(self):
        matrix = np.random.rand(3, 3)
        actual = ShapeAdaptive._validate_bandwidth_matrix(matrix)
        self.assertIsNone(actual)

    def test_evaluate_bandwidth_matrix_1(self):
        matrix = np.random.rand(3)
        try:
            ShapeAdaptive._validate_bandwidth_matrix(matrix)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_evaluate_bandwidth_matrix_2(self):
        matrix = np.random.rand(3, 4, 5)
        try:
            ShapeAdaptive._validate_bandwidth_matrix(matrix)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_evaluate_bandwidth_matrix_3(self):
        matrix = np.random.rand(3, 6)
        try:
            ShapeAdaptive._validate_bandwidth_matrix(matrix)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_evaluate_bandwidth_matrix_4(self):
        matrix = np.random.rand(6, 3)
        try:
            ShapeAdaptive._validate_bandwidth_matrix(matrix)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')
