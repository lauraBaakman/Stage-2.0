from unittest import TestCase

import numpy as np

from datasets.simulated.simulateddataset import SimulatedDataSet
import datasets.simulated.components as components


class SimulatedDataSetsForTest(SimulatedDataSet):

    def _init_components(self):
        return {
            'a': {
                'component': components.UniformRandomNoise(
                    minimum_value=0,
                    maximum_value=20
                ),
                'num elements': 10,
            },
            'b': {
                'component': components.UniformRandomNoise(
                    minimum_value=10,
                    maximum_value=50
                ),
                'num elements': 5,
            },
        }


class TestSimulatedDataSet(TestCase):

    def setUp(self):
        super().setUp()
        self._data_set = SimulatedDataSetsForTest()

    def test__compute_patterns_shape(self):
        actual_patterns = self._data_set.patterns
        (actual_num_patterns, actual_dimension) = actual_patterns.shape

        expected_num_patterns = 15
        expected_dimension = 3

        self.assertEqual(actual_num_patterns, expected_num_patterns)
        self.assertEqual(actual_dimension, expected_dimension)

    def test__compute_patterns_range(self):
        actual_patterns = self._data_set.patterns

        first_component_patterns = actual_patterns[0:10]
        self._in_range(first_component_patterns, 0, 20)

        second_component_patterns = actual_patterns[10:15]
        self._in_range(second_component_patterns, 10, 50)

    # noinspection PyTypeChecker
    def _in_range(self, values, min_value, max_value):
        self.assertTrue(np.all(values >= min_value))
        self.assertTrue(np.all(values <= max_value))

    def test__compute_densities_shape(self):
        actual_densities = self._data_set.densities
        (actual_num_densities, ) = actual_densities.shape

        expected_num_densities = 15
        self.assertEqual(actual_num_densities, expected_num_densities)

    def test__compute_densities_values(self):
        patterns = np.array([
            [0, 0, 0], #In neither component
            [5, 5, 5],  # In the first component
            [40, 40, 40],  # In the second component
            [15, 15, 15] #In both components
        ])

        expected_densities = np.array([
            0.0,
            0.000125000000000,
            0.000015625000000,
            0.000000001953125,
        ])
        actual_densities = self._data_set._compute_densities(patterns)