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

    def test_define_and_validate_input(self):
        dimension = 3
        num_patterns = 10
        patterns = np.random.rand(num_patterns, dimension)
        bandwidth_matrix = np.random.rand(dimension, dimension)
        actual_patterns = ShapeAdaptiveKernel(bandwidth_matrix)._define_and_validate_input(
            patterns
        )
        expected_patterns = patterns

        np.testing.assert_array_equal(actual_patterns, expected_patterns)


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
