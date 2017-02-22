from unittest import TestCase

import numpy as np

from kde.kernels.standardGaussian import StandardGaussian, _StandardGaussian_C, _StandardGaussian_Python


class TestStandardGaussian(TestCase):
    def test_evaluate_default_implementation(self):
        x = np.array([0.5, 0.5])
        actual = StandardGaussian().evaluate(x)
        expected = np.array([0.123949994309653])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_alternative_implementation(self):
        x = np.array([0.5, 0.5])
        actual = StandardGaussian(implementation=_StandardGaussian_Python).evaluate(x)
        expected = np.array([0.123949994309653])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_to_C_enum(self):
        expected = 1
        actual = StandardGaussian().to_C_enum()
        self.assertEqual(expected, actual)

    def test_scaling_factor(self):
        self.fail()

class StandardGaussianImpAbstractTest(object):
    def test_evaluate_2D_1(self):
        x = np.array([0.5, 0.5])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.123949994309653])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_2D_2(self):
        x = np.array([-0.75, -0.5])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.106020048452543])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_2D_3(self):
        x = np.array([0, 0])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.159154943091895])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_2D_multiple(self):
        x = np.array([[0, 0], [-0.75, -0.5], [0.5, 0.5]])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.159154943091895, 0.106020048452543, 0.123949994309653])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_3D_1(self):
        x = np.array([0.5, 0.5, 0.5])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.043638495249061])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_3D_2(self):
        x = np.array([-0.75, -0.5, 0.1])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.042084928316873])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_3D_3(self):
        x = np.array([0, 0, 0])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.063493635934241])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_3D_multiple(self):
        x = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [-0.75, -0.5, 0.1]])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.063493635934241, 0.043638495249061, 0.042084928316873])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_scaling_factor(self):
        self.fail()

class Test_StandardGaussian_Python(StandardGaussianImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._kernel_class = _StandardGaussian_Python


class Test_StandardGaussian_C(StandardGaussianImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._kernel_class = _StandardGaussian_C
