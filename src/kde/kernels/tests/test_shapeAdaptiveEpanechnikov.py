from unittest import TestCase

import numpy as np

from kde.kernels.shapeadaptiveepanechnikov import ShapeAdaptiveEpanechnikov, \
    _ShapeAdaptiveEpanechnikov_C, _ShapeAdaptiveEpanechnikov_Python


class TestShapeAdaptiveEpanechnikov(TestCase):
    def test_to_C_enum(self):
        actual = ShapeAdaptiveEpanechnikov.to_C_enum()
        expected = 3
        self.assertEqual(actual, expected)

    def test_default_implementation(self):
        self.fail("not implemented")

    def test_alternative_implementation(self):
        self.fail("not implemented")


class ShapeAdaptiveGaussianImpAbstractTest(object):

    def test_to_C_enum(self):
        actual = ShapeAdaptiveEpanechnikov.to_C_enum()
        expected = 3
        self.assertEqual(actual, expected)

    def test_evalute_single_pattern_l_eq_one(self):
        self.fail('Not Implemented')

    def test_evalute_single_pattern_l_neq_one(self):
        self.fail('Not Implemented')

    def test_evalute_multiple_patterns_l_eq_one(self):
        self.fail('Not Implemented')

    def test_evalute_multiple_patterns_l_neq_one(self):
        self.fail('Not Implemented')

class Test_ShapeAdaptiveEpanechnikov_Python(ShapeAdaptiveGaussianImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._kernel_class = _ShapeAdaptiveEpanechnikov_Python


class Test_ShapeAdaptiveEpanechnikov_C(ShapeAdaptiveGaussianImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._kernel_class = _ShapeAdaptiveEpanechnikov_C