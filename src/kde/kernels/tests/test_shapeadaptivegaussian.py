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
        raise NotImplementedError()

    def test_alternative_implementation(self):
        raise NotImplementedError()

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

    def setUp(self):
        super().setUp()

    def test_dimension(self):
        raise NotImplementedError()

    def test_validate_patterns_0(self):
        # Multiple patterns, correct dimension
        raise NotImplementedError()

    def test_validate_patterns_1(self):
        # Single pattern, correct dimension
        raise NotImplementedError()

    def test_validate_patterns_2(self):
        # Multiple patterns, wrong dimension
        raise NotImplementedError()

    def test_validate_patterns_3(self):
        # Single patterns, wrong dimension
        raise NotImplementedError()

    def test_evaluate_0(self):
        # Single pattern, local bandwidth = 1
        raise NotImplementedError()

    def test_evaluate_1(self):
        # Multiple patterns, local bandwidth = 1
        raise NotImplementedError()

    def test_evaluate_2(self):
        # Single patterns, local bandwidth \neq 1
        raise NotImplementedError()

    def test_evaluate_3(self):
        # Multiple patterns, local bandwidth \neq 1
        raise NotImplementedError()


class ShapeAdaptiveGaussianImpAbstractTest(object):

    def test_evaluate_0(self):
        # Single Pattern, local bandwidth = 1
        raise NotImplementedError()

    def test_evaluate_1(self):
        # Single Pattern, local bandwidth \neq 1
        raise NotImplementedError()

    def test_evaluate_2(self):
        # Multiple Patterns, local bandwidth = 1
        raise NotImplementedError()

    def test_evaluate_3(self):
        # Multiple Patterns, local bandwidth neq 1
        raise NotImplementedError()


class Test_ShapeAdaptiveGaussian_Python(ShapeAdaptiveGaussianImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._kernel_class = _ShapeAdaptiveGaussian_Python


class Test_ShapeAdaptive_C(ShapeAdaptiveGaussianImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._kernel_class = _ShapeAdaptiveGaussian_C