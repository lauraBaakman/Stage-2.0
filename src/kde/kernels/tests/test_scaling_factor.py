from unittest import TestCase

import numpy as np

import kde.kernels.scaling as scaling


class TestScaling_factor(TestCase):
    def _scaling_factor_not_needed(self, dimension, implementation):
        h = 0.5
        covariance_matrix = h * np.identity(dimension)
        actual = implementation(general_bandwidth=h,
                                covariance_matrix=covariance_matrix)
        expected = 1.0
        self.assertAlmostEqual(actual, expected)

    def _scaling_factor_3_auxiliary(self, dimension, implementation):
        h = 0.5
        covariance_matrix = np.identity(dimension)
        actual = implementation(general_bandwidth=h,
                                covariance_matrix=covariance_matrix)
        expected = h
        self.assertAlmostEqual(actual, expected)

    def _scaling_factor_2_auxiliary(self, implementation):
        general_bandwidth = 0.5
        covariance_matrix = np.array([[0.082389650806275, 0.003023884755706, -0.000211227775848],
                                      [0.003023884755706, 0.083836642887593, -0.002950447026358],
                                      [-0.000211227775848, -0.002950447026358, 0.086068346308987]])
        expected = 8.431798865555362e+02
        actual = implementation(general_bandwidth=general_bandwidth,
                                covariance_matrix=covariance_matrix)
        self.assertAlmostEqual(expected, actual)

    def test_scaling_factor_0_python(self):
        self._scaling_factor_not_needed(2, scaling._scaling_factor_python)

    def test_scaling_factor_1_python(self):
        self._scaling_factor_not_needed(5, scaling._scaling_factor_python)

    def test_scaling_factor_2_python(self):
        self._scaling_factor_2_auxiliary(scaling._scaling_factor_python)

    def test_scaling_factor_3_python(self):
        self._scaling_factor_3_auxiliary(scaling._scaling_factor_python)

    def test_scaling_factor_0_c(self):
        self._scaling_factor_not_needed(2, scaling._scaling_factor_c)

    def test_scaling_factor_1_c(self):
        self._scaling_factor_not_needed(5, scaling._scaling_factor_c)

    def test_scaling_factor_2_c(self):
        self._scaling_factor_2_auxiliary(scaling._scaling_factor_c)

    def test_scaling_factor_3_c(self):
        self._scaling_factor_3_auxiliary(scaling._scaling_factor_c)

    def test_scaling_factor_c(self):
        h = 0.5
        dimension = 5
        covariance_matrix = h * np.identity(dimension)
        actual = scaling.scaling_factor(general_bandwidth=h,
                                        covariance_matrix=covariance_matrix,
                                        implementation=scaling._scaling_factor_c)
        expected = 1.0
        self.assertAlmostEqual(actual, expected)

    def test_scaling_factor_python(self):
        dimension = 5
        h = 0.5
        covariance_matrix = h * np.identity(dimension)
        actual = scaling.scaling_factor(general_bandwidth=h,
                                        covariance_matrix=covariance_matrix,
                                        implementation=scaling._scaling_factor_python)
        expected = 1.0
        self.assertAlmostEqual(actual, expected)