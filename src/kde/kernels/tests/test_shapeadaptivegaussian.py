from unittest import TestCase, skip

import numpy as np

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

    def test_evaluate_bandwidth_matrix(self):
        raise NotImplementedError()


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