from unittest import TestCase

import numpy as np

import kde


class TestParzen(TestCase):

    def setUp(self):
        super().setUp()
        self.data = {
            'patterns': np.array([
                [2, 3],
                [1, 3],
                [-2, 2]
            ]),
            'parzen_estimated_densities': np.array([
                0.591741568117374,
                0.591741568117374,
                0.589462752192205
            ]),
            'window_width': 0.3,
            'dimension': 2
        }

    def test_estimate(self):
        kernel_shape = self.data['window_width'] * self.data['window_width'] * np.identity(self.data['dimension'])
        kernel = kde.kernels.Gaussian(covariance_matrix=kernel_shape)
        estimator = kde.Parzen(
            dimension=self.data['dimension'],
            window_width=self.data['window_width'],
            kernel=kernel
        )
        actual = estimator.estimate(xi_s=self.data['patterns'])
        expected = self.data['parzen_estimated_densities']
        np.testing.assert_array_almost_equal(actual, expected)
