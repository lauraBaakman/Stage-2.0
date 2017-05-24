from unittest import TestCase, skip

import numpy as np

from kde.kernels.epanechnikov import Epanechnikov, _Epanechnikov_Python, _Epanechnikov_C


class TestEpanechnikov(TestCase):
    def test_evaluate_default_implementation(self):
        x = np.array([5])
        actual = Epanechnikov().evaluate(x)
        expected = np.array([0])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_alternative_implementation(self):
        x = np.array([5])
        actual = Epanechnikov(implementation=_Epanechnikov_Python).evaluate(x)
        expected = np.array([0])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_to_C_enum(self):
        expected = 2
        actual = Epanechnikov().to_C_enum()
        self.assertEqual(expected, actual)


class EpanechnikovImpAbstractTest(object):

    def test_evaluate_1D_1(self):
        x = np.array([5.0])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_1D_2(self):
        x = np.array([-0.5])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.577297133446585])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_1D_3(self):
        x = np.array([0.5])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.577297133446585])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_1D_4(self):
        x = np.array([[5], [0.5]])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0, 0.577297133446585])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_2D_1(self):
        x = np.array([[5, 0.5]])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_2D_2(self):
        x = np.array([0.5, 0.5])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.287224936361155])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_2D_3(self):
        x = np.array([[0.3, 0.5], [0.5, 0.5]])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.462693261119970, 0.287224936361155])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_3D(self):
        x = np.array([0.5, 0.5, 0.5])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.014022336779338])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_4D(self):
        x = np.array([0.3, 0.5, 0.5, 0.5])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_4D_2(self):
        x = np.array([0.2, 0.2, 0.1, 0.5])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.579914367185099])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_5D_1(self):
        x = np.array([0.3, 0.5, 0.5, 0.5, 0.4])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_5D_2(self):
        x = np.array([0.2, 0.2, 0.1, 0.5, 0.01])
        actual = self._kernel_class().evaluate(x)
        expected = np.array([0.726488328774824])
        np.testing.assert_array_almost_equal(actual, expected)

class TestEpanechnikov_Python(EpanechnikovImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
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
        super().setUp()
        self._kernel_class = _Epanechnikov_C
