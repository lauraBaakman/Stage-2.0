from unittest import TestCase

import numpy as np

from kde.kernels.gaussian import Gaussian, _Gaussian_C, _Gaussian_Python


class TestGaussian(TestCase):
    def test_evaluate_default_implementation(self):
        x = np.array([0.5, 0.5])
        actual = Gaussian().evaluate(x)
        expected = np.array([0.123949994309653])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_alternative_implementation(self):
        x = np.array([0.5, 0.5])
        actual = Gaussian(implementation=_Gaussian_Python).evaluate(x)
        expected = np.array([0.123949994309653])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_to_C_enum(self):
        expected = 1
        actual = Gaussian().to_C_enum()
        self.assertEqual(expected, actual)

    def test_radius(self):
        bandwidth = 2.5
        actual = Gaussian().radius(bandwidth)
        expected = float("inf")
        self.assertAlmostEqual(actual, expected)


class GaussianImpAbstractTest(object):

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


class Test_Gaussian_Python(GaussianImpAbstractTest, TestCase):

    def setUp(self):
        super(Test_Gaussian_Python, self).setUp()
        self._kernel_class = _Gaussian_Python


class Test_Gaussian_C(GaussianImpAbstractTest, TestCase):

    def setUp(self):
        super(Test_Gaussian_C, self).setUp()
        self._kernel_class = _Gaussian_C
