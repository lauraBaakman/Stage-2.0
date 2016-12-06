from unittest import TestCase

import numpy as np

import kde.modifeidbreiman as mbe
from kde.kernels.testKernel import TestKernel
from kdeUtils.grid import Grid


class Test_MBEstimator(TestCase):

    def setUp(self):
        super().setUp()
        self.data = {
            'xi_s': np.array([
                [2, 2],
                [1, 3],
                [2, 3]
            ]),
            'x_s': np.array([
                [1, 2],
                [1, 2]
            ]),
            'MBE_final_densities': np.array([
                - 352 / 3,
                - 352 / 3
            ]),
            'window_widths':np.array([0.5, 0.25, 0.5]),
            'general_window_width': 0.5,
            'dimension': 2,
            'kernel': TestKernel()
        }

    def test_estimate(self):
        estimator = mbe._MBEstimator(xi_s=self.data['xi_s'],
                                     x_s=self.data['x_s'],
                                     dimension=self.data['dimension'],
                                     kernel=self.data['kernel'],
                                     local_bandwidths=self.data['window_widths'],
                                     general_bandwidth=self.data['general_window_width'])
        actual = estimator.estimate()
        expected = self.data['MBE_final_densities']
        np.testing.assert_array_almost_equal(actual, expected)

    def test_estimate_on_grid(self):
        xi_s = Grid((0, 3), number_of_grid_points=2).grid_points
        x_s = xi_s
        dimension = 1
        kernel = TestKernel()
        local_bandwidths = np.array([1, 2])
        general_bandwidth = 0.5
        estimator = mbe._MBEstimator(xi_s=xi_s, x_s=x_s,
                                     dimension=dimension, kernel=kernel,
                                     local_bandwidths=local_bandwidths, general_bandwidth=general_bandwidth)
        actual = estimator.estimate()
        expected = np.array([-1.5, 6])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_estimate_with_xs(self):
        xi_s = Grid((0, 3), number_of_grid_points=2).grid_points
        x_s = np.array([[2]])
        dimension = 1
        kernel = TestKernel()
        local_bandwidths = np.array([1, 2])
        general_bandwidth = 0.5
        estimator = mbe._MBEstimator(xi_s=xi_s, x_s=x_s,
                                     dimension=dimension, kernel=kernel,
                                     local_bandwidths=local_bandwidths, general_bandwidth=general_bandwidth)
        actual = estimator.estimate()
        expected = np.array([3.5])
        np.testing.assert_array_almost_equal(actual, expected)
