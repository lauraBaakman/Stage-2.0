from unittest import TestCase

import numpy as np

from datasets.simulated.components import UniformRandomNoise


class TestUniformRandomNoise(TestCase):

    def setUp(self):
        super().setUp()
        self._min_value = 2
        self._max_value = 4
        self._component = UniformRandomNoise(minimum_value=self._min_value,
                                             maximum_value=self._max_value)

    def test_patterns_shape_0(self):
        expected_num_patterns = 1
        expected_dimension = 3

        actual_patterns = self._component.patterns(num_patterns=expected_num_patterns)
        (actual_num_patterns, actual_dimension) = actual_patterns.shape

        self.assertEqual(actual_num_patterns, expected_num_patterns)
        self.assertEqual(actual_dimension, expected_dimension)

    def test_patterns_shape_1(self):
        expected_num_patterns = 15
        expected_dimension = 3

        actual_patterns = self._component.patterns(num_patterns=expected_num_patterns)
        (actual_num_patterns, actual_dimension) = actual_patterns.shape

        self.assertEqual(actual_num_patterns, expected_num_patterns)
        self.assertEqual(actual_dimension, expected_dimension)

    def test_patterns_values_0(self):
        patterns = self._component.patterns(1)
        self._in_range(patterns.transpose()[0])
        self._in_range(patterns.transpose()[1])
        self._in_range(patterns.transpose()[2])

    def test_patterns_values_1(self):
        patterns = self._component.patterns(10)
        self._in_range(patterns.transpose()[0])
        self._in_range(patterns.transpose()[1])
        self._in_range(patterns.transpose()[2])

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
        patterns = np.array([
            [1.0, 3.0, 3.0],
        ])

        expected_densities = np.array([
            0.0,
        ])
        actual_densities = self._component.densities(patterns)
        np.testing.assert_array_almost_equal(expected_densities, actual_densities)

    def test_densities_values_1(self):
        patterns = np.array([
            [5.0, 3.0, 3.0],
        ])

        expected_densities = np.array([
            0.0,
        ])
        actual_densities = self._component.densities(patterns)
        np.testing.assert_array_almost_equal(expected_densities, actual_densities)

    def test_densities_values_2(self):
        patterns = np.array([
            [3.0, 3.0, 3.0],
        ])

        expected_densities = np.array([
            0.125000000000000,
        ])
        actual_densities = self._component.densities(patterns)
        np.testing.assert_array_almost_equal(expected_densities, actual_densities)

    def test_densities_values_3(self):
        patterns = np.array([
            [1.0, 3.0, 3.0],
            [2.0, 3.0, 3.0],
            [3.5, 3.0, 5.0],
        ])

        expected_densities = np.array([
            0.000000000000000,
            0.125000000000000,
            0.000000000000000,
        ])
        actual_densities = self._component.densities(patterns)
        np.testing.assert_array_almost_equal(expected_densities, actual_densities)

    # noinspection PyTypeChecker
    def _in_range(self, values):
        self.assertTrue(np.all(values >= self._min_value))
        self.assertTrue(np.all(values <= self._max_value))
