from unittest import TestCase

import numpy as np

import kde.kernels.scaling as scaling


class TestScaling_factor(TestCase):
    def _scaling_factor_not_needed(self, dimension, implementation):
        standard_deviation = 0.5
        variance = standard_deviation * standard_deviation
        covariance_matrix = variance * np.identity(dimension)
        actual = implementation(general_bandwidth=standard_deviation,
                                covariance_matrix=covariance_matrix)
        expected = 1.0
        self.assertAlmostEqual(actual, expected)

    def _scaling_factor_2_auxilary(self, implementation):
        general_bandwidth = 0.5
        covariance_matrix = np.array([[0.085990572233016,   0.006485451830366,   0.012006730455573],
                                      [0.006485451830366,   0.081172010593423,   0.007497828233425],
                                      [0.012006730455573,   0.007497828233425,   0.085865865754708]])
        expected = 5.188704957608463
        actual = implementation(general_bandwidth=general_bandwidth,
                                covariance_matrix=covariance_matrix)
        self.assertAlmostEqual(expected, actual)

    def test_scaling_factor_0_python(self):
        self._scaling_factor_not_needed(2, scaling._scaling_factor_python)

    def test_scaling_factor_1_python(self):
        self._scaling_factor_not_needed(5, scaling._scaling_factor_python)

    def test_scaling_factor_2_python(self):
        self._scaling_factor_2_auxilary(scaling._scaling_factor_python)

    def test_scaling_factor_0_c(self):
        self._scaling_factor_not_needed(2, scaling._scaling_factor_c)

    def test_scaling_factor_1_c(self):
        self._scaling_factor_not_needed(5, scaling._scaling_factor_c)

    def test_scaling_factor_2_c(self):
        self._scaling_factor_2_auxilary(scaling._scaling_factor_c)

    def test_scaling_factor_c(self):
        standard_deviation = 0.5
        variance = standard_deviation * standard_deviation
        covariance_matrix = variance * np.identity(3)
        actual = scaling.scaling_factor(general_bandwidth=standard_deviation,
                                        covariance_matrix=covariance_matrix,
                                        implementation=scaling._scaling_factor_c)
        expected = 1.0
        self.assertAlmostEqual(actual, expected)

    def test_scaling_factor_python(self):
        standard_deviation = 0.5
        variance = standard_deviation * standard_deviation
        covariance_matrix = variance * np.identity(3)
        actual = scaling.scaling_factor(general_bandwidth=standard_deviation,
                                        covariance_matrix=covariance_matrix,
                                        implementation=scaling._scaling_factor_python)
        expected = 1.0
        self.assertAlmostEqual(actual, expected)