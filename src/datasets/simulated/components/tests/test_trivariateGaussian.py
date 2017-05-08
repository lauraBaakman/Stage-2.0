from unittest import TestCase

import numpy as np

from datasets.simulated.components import TrivariateGaussian


class TestTrivariateGaussian(TestCase):

    def setUp(self):
        super().setUp()
        self._mean = np.array([50, 50, 50])
        self._covariance = np.diag(np.array([30, 30, 30]))
        self._component = TrivariateGaussian(mean=self._mean,
                                       covariance_matrix=self._covariance)

    def test_patterns_patterns_shape_0(self):
        expected_num_patterns = 1
        expected_dimension = 3

        actual_patterns = self._component.patterns(expected_num_patterns)
        (actual_num_patterns, actual_dimension) = actual_patterns.shape

        self.assertEqual(expected_num_patterns, actual_num_patterns)
        self.assertEqual(expected_dimension, actual_dimension)

    def test_patterns_patterns_shape_1(self):
        expected_num_patterns = 10
        expected_dimension = 3

        actual_patterns = self._component.patterns(expected_num_patterns)
        (actual_num_patterns, actual_dimension) = actual_patterns.shape

        self.assertEqual(expected_num_patterns, actual_num_patterns)
        self.assertEqual(expected_dimension, actual_dimension)

    def test_densities_shape_0(self):
        expected_num_patterns = 1

        patterns = np.random.random((expected_num_patterns, 3))

        actual_densities = self._component.densities(patterns)
        (actual_num_patterns,) = actual_densities.shape
        self.assertEqual(actual_num_patterns, expected_num_patterns)

    def test_densities_shape_1(self):

        expected_num_patterns = 10

        patterns = np.random.random((expected_num_patterns, 3))

        actual_densities = self._component.densities(patterns)
        (actual_num_patterns, ) = actual_densities.shape
        self.assertEqual(actual_num_patterns, expected_num_patterns)

    def test_densities_values_0(self):
        patterns = np.array([[0.6324, 0.9649, 0.8003]])

        expected_densities = np.array([
            0.105040766084006e-55,
        ])
        actual_densities = self._component.densities(patterns)
        np.testing.assert_array_almost_equal(expected_densities, actual_densities)

    def test_densities_values_1(self):
        patterns = np.array([[0.8147, 0.0975, 0.1576],
                             [0.9058, 0.2785, 0.9706],
                             [0.1270, 0.5469, 0.9572],
                             [0.9134, 0.9575, 0.4854],
                             [0.6324, 0.9649, 0.8003]])

        expected_densities = np.array([
            0.011737828503672e-55,
            0.070254716194359e-55,
            0.029642020542178e-55,
            0.098033561414839e-55,
            0.105040766084006e-55,
        ])
        actual_densities = self._component.densities(patterns)
        np.testing.assert_array_almost_equal(expected_densities, actual_densities)