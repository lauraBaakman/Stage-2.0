from unittest import TestCase
import warnings

import numpy as np

from kde.kernels import StandardGaussian


class TestStandardGaussian(TestCase):

    def setUp(self):
        super().setUp()
        self.data = {
            'patterns': [
                np.array([0.5, 0.5]),
                np.array([-0.75, -0.5]),
                np.array([0, 0]),
            ],
            'densities': [
                0.123949994309653,
                0.106020048452543,
                0.159154943091895,
            ],
            'dimension': 2,
        }

    def test_center_get(self):
        kernel = StandardGaussian(dimension=self.data['dimension'])
        with warnings.catch_warnings(record=True) as warning:
            warnings.simplefilter("always")
            actual = kernel.center
            assert issubclass(warning[-1].category, UserWarning)

    def test_center_set(self):
        kernel = StandardGaussian(dimension=self.data['dimension'])
        with warnings.catch_warnings(record=True) as warning:
            warnings.simplefilter("always")
            kernel.center = np.array([0.5 , 0.5])
            assert issubclass(warning[-1].category, UserWarning)

    def test_shape_get(self):
        kernel = StandardGaussian(dimension=self.data['dimension'])
        with warnings.catch_warnings(record=True) as warning:
            warnings.simplefilter("always")
            actual = kernel.shape
            assert issubclass(warning[-1].category, UserWarning)

    def test_shape_set(self):
        kernel = StandardGaussian(dimension=self.data['dimension'])
        with warnings.catch_warnings(record=True) as warning:
            warnings.simplefilter("always")
            kernel.shape = np.array([[0.5, 0.5], [0.5, 1.5]])
            assert issubclass(warning[-1].category, UserWarning)

    def test_evaluate(self):
        kernel = StandardGaussian(dimension=self.data['dimension'])
        for pattern, expected in zip(self.data['patterns'], self.data['densities']):
            actual = kernel.evaluate(pattern)
            self.assertAlmostEqual(actual, expected)
