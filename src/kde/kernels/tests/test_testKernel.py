from unittest import TestCase,skip

import numpy as np

from kde.kernels.testKernel import TestKernel, _TestKernel_C, _TestKernel_Python


class TestTestKernel(TestCase):
    def test_evaluate_default_implementation(self):
        x = np.array([[2, 3], [3, 4], [4, 5]])
        actual = TestKernel().evaluate(x)
        expected = np.array([2.5, 3.5, 4.5])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_alternative_implementation(self):
        x = np.array([[2, 3], [3, 4], [4, 5]])
        actual = TestKernel(implementation=_TestKernel_C).evaluate(x)
        expected = np.array([2.5, 3.5, 4.5])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_to_C_enum(self):
        expected = 0
        actual = TestKernel().to_C_enum()
        self.assertEqual(expected, actual)

    def test_scaling_factor(self):
        expected = 0.5
        actual = TestKernel().scaling_factor()
        self.assertEqual(actual, expected)


class TestKernelImpAbstractTest(object):
    def test_evaluate_1(self):
        x = np.array([[2, 3], [3, 4], [4, 5]])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([2.5, 3.5, 4.5])
        np.testing.assert_array_equal(actual, expected)

    def test_evaluate_2(self):
        x = np.array([[-2, -3], [-3, -4], [-4, -5]])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([2.5, 3.5, 4.5])
        np.testing.assert_array_equal(actual, expected)

    def test_scaling_factor(self):
        expected = 0.5
        actual = self._kernel_class().scaling_factor()
        self.assertEqual(actual, expected)


class TestTestKernel_C(TestKernelImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
        self._kernel_class = _TestKernel_C

    def test_scaling_factor(self):
        self.skipTest("The function scaling_factor has not been implemented for the Test kernel.")

class TestTestKernel_Python(TestKernelImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
        self._kernel_class = _TestKernel_Python

