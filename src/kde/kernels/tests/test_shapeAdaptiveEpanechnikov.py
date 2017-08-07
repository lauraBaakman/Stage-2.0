from unittest import TestCase

import numpy as np

from kde.kernels.shapeadaptiveepanechnikov import ShapeAdaptiveEpanechnikov, \
    _ShapeAdaptiveEpanechnikov_C, _ShapeAdaptiveEpanechnikov_Python


class TestShapeAdaptiveEpanechnikov(TestCase):
    def test_to_C_enum(self):
        actual = ShapeAdaptiveEpanechnikov.to_C_enum()
        expected = 3
        self.assertEqual(actual, expected)

    def test_default_implementation_single_pattern_l_eq_1(self):
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        x = np.array([0.05, 0.05, 0.05])
        expected = 0.013288829111394
        actual = ShapeAdaptiveEpanechnikov(H).evaluate(x)
        self.assertAlmostEqual(actual, expected)

    def test_alternative_implementation_single_pattern_l_eq_1(self):
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        x = np.array([0.05, 0.05, 0.05])
        expected = 0.013288829111394
        actual = ShapeAdaptiveEpanechnikov(
            H,
            implementation=_ShapeAdaptiveEpanechnikov_Python
        ).evaluate(x)
        self.assertAlmostEqual(actual, expected)


class ShapeAdaptiveGaussianImpAbstractTest(object):

    def test_to_C_enum(self):
        actual = ShapeAdaptiveEpanechnikov.to_C_enum()
        expected = 3
        self.assertEqual(actual, expected)

    def test_evalute_single_pattern_l_eq_one(self):
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        x = np.array([0.05, 0.05, 0.05])
        actual = self._kernel_class(H).evaluate(x)
        expected = 0.013288829111394
        self.assertAlmostEqual(actual, expected)

    def test_evalute_multiple_patterns_l_eq_one(self):
        H = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        x = np.array([
            [0.05, 0.05, 0.05],
            [0.02, 0.03, 0.04],
            [0.04, 0.05, 0.03]
        ])
        expected = np.array([
            0.013288829111394,
            0.013324995545631,
            0.013307012420120
        ])
        actual = self._kernel_class(H).evaluate(x)
        np.testing.assert_array_almost_equal(actual, expected)


class Test_ShapeAdaptiveEpanechnikov_Python(ShapeAdaptiveGaussianImpAbstractTest, TestCase):

    def setUp(self):
        super(Test_ShapeAdaptiveEpanechnikov_Python, self).setUp()
        self._kernel_class = _ShapeAdaptiveEpanechnikov_Python


class Test_ShapeAdaptiveEpanechnikov_C(ShapeAdaptiveGaussianImpAbstractTest, TestCase):

    def setUp(self):
        super(Test_ShapeAdaptiveEpanechnikov_C, self).setUp()
        self._kernel_class = _ShapeAdaptiveEpanechnikov_C
