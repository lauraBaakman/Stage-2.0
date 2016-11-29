from unittest import TestCase
import warnings

import numpy as np

# from kde.kernels import StandardGaussian_New as StandardGaussian
from kde.kernels import StandardGaussian as StandardGaussian


class TestStandardGaussian(TestCase):

    def setUp(self):
        super().setUp()
        self.data_2D = {
            'patterns': [
                np.array([0.5, 0.5]),
                np.array([-0.75, -0.5]),
                np.array([0, 0]),gi
            ],
            'densities': [
                0.123949994309653,
                0.106020048452543,
                0.159154943091895,
            ],
            'dimension': 2,
        }
        self.data_3D = {
            'patterns': [
                np.array([0.5, 0.5, 0.5]),
                np.array([-0.75, -0.5, 0.1]),
                np.array([0, 0, 0]),
            ],
            'densities': [
                0.043638495249061,
                0.042084928316873,
                0.063493635934241,
            ],
            'dimension': 3,
        }

    def test_center_get(self):
        kernel = StandardGaussian(dimension=self.data_2D['dimension'])
        with warnings.catch_warnings(record=True) as warning:
            warnings.simplefilter("always")
            actual = kernel.center
            assert issubclass(warning[-1].category, UserWarning)

    def test_center_set(self):
        kernel = StandardGaussian(dimension=self.data_2D['dimension'])
        with warnings.catch_warnings(record=True) as warning:
            warnings.simplefilter("always")
            kernel.center = np.array([0.5 , 0.5])
            assert issubclass(warning[-1].category, UserWarning)

    def test_shape_get(self):
        kernel = StandardGaussian(dimension=self.data_2D['dimension'])
        with warnings.catch_warnings(record=True) as warning:
            warnings.simplefilter("always")
            actual = kernel.shape
            assert issubclass(warning[-1].category, UserWarning)

    def test_shape_set(self):
        kernel = StandardGaussian(dimension=self.data_2D['dimension'])
        with warnings.catch_warnings(record=True) as warning:
            warnings.simplefilter("always")
            kernel.shape = np.array([[0.5, 0.5], [0.5, 1.5]])
            assert issubclass(warning[-1].category, UserWarning)

    def test_evaluate_2D(self):
        kernel = StandardGaussian(dimension=self.data_2D['dimension'])
        for pattern, expected in zip(self.data_2D['patterns'], self.data_2D['densities']):
            actual = kernel.evaluate(pattern)
            self.assertAlmostEqual(actual, expected)

    def test_evaluate_3D(self):
        kernel = StandardGaussian(dimension=self.data_3D['dimension'])
        for pattern, expected in zip(self.data_3D['patterns'], self.data_3D['densities']):
            actual = kernel.evaluate(pattern)
            self.assertAlmostEqual(actual, expected)

    def test_evaluate_2D_multiple(self):
        kernel = StandardGaussian(dimension=self.data_2D['dimension'])
        patterns = np.matrix(self.data_2D['patterns'])
        actual = kernel.evaluate(patterns)
        expected = np.array(self.data_2D['densities'])
        np.testing.assert_array_almost_equal(actual, expected)
