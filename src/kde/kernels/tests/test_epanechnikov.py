from unittest import TestCase

import numpy as np

from kde.kernels.epanechnikov import Epanechnikov
from kde.kernels.epanechnikov import _Epanechnikov_Python, _Epanechnikov_C


class TestEpanechnikov(TestCase):
    def test_evaluate_default_implementation(self):
        x = np.array([5])
        actual = Epanechnikov().evaluate(x)
        expected = np.array([0])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_alternative_implementation(self):
        x = np.array([5.0])
        actual = Epanechnikov(implementation=_Epanechnikov_Python).evaluate(x)
        expected = np.array([0])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_to_C_enum(self):
        expected = 2
        actual = Epanechnikov().to_C_enum()
        self.assertEqual(expected, actual)

    def test_radius(self):
        bandwidth = 2.5
        actual = Epanechnikov().radius(bandwidth)
        expected = 2.5 * np.sqrt(5)
        self.assertAlmostEqual(actual, expected)


class EpanechnikovImpAbstractTest(object):

    def test_evaluate_1D_1(self):
        x = np.array([5.0])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_1D_2(self):
        x = np.array([-0.5])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.318639686793720])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_1D_3(self):
        x = np.array([0.5])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.318639686793720])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_1D_4(self):
        x = np.array([[5], [0.5]])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.0, 0.318639686793720])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_2D_1(self):
        x = np.array([[5, 0.5]])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_2D_2(self):
        x = np.array([0.5, 0.5])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.114591559026165])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_2D_3(self):
        x = np.array([[0.3, 0.5], [0.5, 0.5]])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.118665925569317, 0.114591559026165])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_2D_4(self):
        x = np.array([
            [+0.125,  0.175],
            [-0.125,  0.175],
            [-0.375,  0.175],
            [-0.625,  0.175],
            [+0.125, -0.075],
            [-0.125, -0.075],
            [-0.375, -0.075],
            [-0.625, -0.075],
            [+0.125, -0.325],
            [-0.125, -0.325],
            [-0.375, -0.325],
            [-0.625, -0.325],
            [+0.125, -0.575],
            [-0.125, -0.575],
            [-0.375, -0.575],
            [-0.625, -0.575],
        ])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([
            0.126146207895,
            0.126146207895,
            0.122963109033,
            0.116596911309,
            0.126782827667,
            0.126782827667,
            0.123599728805,
            0.117233531081,
            0.124236348578,
            0.124236348578,
            0.121053249716,
            0.114687051992,
            0.118506770626,
            0.118506770626,
            0.115323671764,
            0.108957474041,
        ])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_3D(self):
        x = np.array([0.5, 0.5, 0.5])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.045374862142845])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_4D(self):
        x = np.array([0.3, 0.5, 0.5, 0.5])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.020231813949702])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_4D_2(self):
        x = np.array([0.2, 0.2, 0.1, 0.5])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.022663522357118])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_5D_1(self):
        x = np.array([0.3, 0.5, 0.5, 0.5, 0.4])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.009515564275770])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_5D_2(self):
        x = np.array([0.2, 0.2, 0.1, 0.5, 0.01])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.011085394492165])
        np.testing.assert_array_almost_equal(actual, expected)


class TestEpanechnikov_Python(EpanechnikovImpAbstractTest, TestCase):
    def setUp(self):
        super(TestEpanechnikov_Python, self).setUp()
        self._kernel_class = _Epanechnikov_Python

    def test___unit_sphere_volume_1D(self):
        dimension = 1
        expected = 2
        actual = self._kernel_class()._unit_sphere_volume(dimension)
        self.assertAlmostEqual(actual, expected)

    def test___unit_sphere_volume_2D(self):
        dimension = 2
        expected = 3.141592653589793
        actual = self._kernel_class()._unit_sphere_volume(dimension)
        self.assertAlmostEqual(actual, expected)

    def test___unit_sphere_volume_3D(self):
        dimension = 3
        expected = 4.188790204786391
        actual = self._kernel_class()._unit_sphere_volume(dimension)
        self.assertAlmostEqual(actual, expected)

    def test___unit_sphere_volume_4D(self):
        dimension = 4
        expected = 4.934802200544679
        actual = self._kernel_class()._unit_sphere_volume(dimension)
        self.assertAlmostEqual(actual, expected)

    def test___unit_sphere_volume_5D(self):
        dimension = 5
        expected = 5.263789013914325
        actual = self._kernel_class()._unit_sphere_volume(dimension)
        self.assertAlmostEqual(actual, expected)


class TestEpanechnikov_C(EpanechnikovImpAbstractTest, TestCase):
    def setUp(self):
        super(TestEpanechnikov_C, self).setUp()
        self._kernel_class = _Epanechnikov_C
