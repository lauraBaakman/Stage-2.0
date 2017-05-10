from unittest import TestCase
from collections import OrderedDict
from io import StringIO

import numpy as np

from datasets.simulated.simulateddataset import SimulatedDataSet
import datasets.simulated.components as components


component_a = {
    'component': components.UniformRandomNoise(
        minimum_value=0,maximum_value=20, dimension=3),
    'num elements': 3,
}
component_b = {
    'component': components.UniformRandomNoise(
        minimum_value=10, maximum_value=50, dimension=3),
    'num elements': 2,
}


class SimulatedDataSetForTests(SimulatedDataSet):

    def __init__(self):
        super(SimulatedDataSetForTests, self).__init__()

    def _init_components(self):
        self._components['a'] = component_a
        self._components['b'] = component_b


class TestSimulatedDataSet(TestCase):

    def setUp(self):
        super().setUp()
        self._components = OrderedDict()
        self._components['a'] = component_a
        self._components['b'] = component_b

    def test_components_lengths(self):
        set = SimulatedDataSetForTests()
        actual = set.components_lengths
        expected = [3, 2]
        self.assertEqual(actual, expected)

    def test__compute_patterns_shape(self):
        actual_patterns = SimulatedDataSet._compute_patterns(None, self._components)
        (actual_num_patterns, actual_dimension) = actual_patterns.shape

        expected_num_patterns = 5
        expected_dimension = 3

        self.assertEqual(actual_num_patterns, expected_num_patterns)
        self.assertEqual(actual_dimension, expected_dimension)

    def test__compute_patterns_range(self):
        actual_patterns = SimulatedDataSet._compute_patterns(None, self._components)

        first_component_patterns = actual_patterns[0:3]
        self._in_range(first_component_patterns, 0, 20)

        second_component_patterns = actual_patterns[3:5]
        self._in_range(second_component_patterns, 10, 50)

    # noinspection PyTypeChecker
    def _in_range(self, values, min_value, max_value):
        self.assertTrue(np.all(values >= min_value))
        self.assertTrue(np.all(values <= max_value))

    def test_fixed_seed(self):
        set1 = SimulatedDataSetForTests()
        set2 = SimulatedDataSetForTests()

        self.assertEqual(set1, set2)
        np.testing.assert_array_equal(set1.patterns, set2.patterns)
        np.testing.assert_array_equal(set1.densities, set2.densities)

    def test__compute_densities_shape(self):
        patterns = np.array([
            [-10, -10, -10],
            [5, 5, 5],
            [40, 40, 40],
            [15, 15, 15]
        ])

        actual_densities = SimulatedDataSet._compute_densities(None, self._components, patterns)
        (actual_num_densities, ) = actual_densities.shape

        expected_num_densities = 4
        self.assertEqual(actual_num_densities, expected_num_densities)

    def test__compute_densities_values(self):
        patterns = np.array([
            [-10, -10, -10],    # In neither component
            [5, 5, 5],          # In the first component
            [40, 40, 40],       # In the second component
            [15, 15, 15]        # In both components
        ])
        actual_densities = SimulatedDataSet._compute_densities(None, self._components, patterns)
        expected_densities = np.array([
            0.0,
            0.0000625,
            0.0000078125,
            0.0000703125
        ])
        np.testing.assert_array_almost_equal(expected_densities, actual_densities)

    def test_to_file(self):
        data_set = SimulatedDataSetForTests()

        expected_output = ("5 3\n"
                           "10.97627008,  14.30378733,  12.05526752\n"
                           "10.89766366,   8.47309599,  12.91788226\n"
                           "8.75174423,  17.83546002,  19.27325521\n"
                           "25.33766075,  41.66900152,  31.15579679\n"
                           "32.72178244,  47.02386553,  12.84144233\n"
                           "7.03125000e-05\n"
                           "6.25000000e-05\n"
                           "6.25000000e-05\n"
                           "7.81250000e-06\n"
                           "7.81250000e-06\n")

        # See: http://stackoverflow.com/a/3945057/1357229
        actual_file_buffer = StringIO()
        data_set.to_file(actual_file_buffer)
        actual_file_buffer.seek(0)
        actual_output = actual_file_buffer.read()

        self.assertEqual(actual_output, expected_output)

    def test__header_to_file(self):
        data_set = SimulatedDataSetForTests()

        expected_output = ("5 3\n"
                           "3 2\n")

        actual_file_buffer = StringIO()
        data_set._header_to_file(actual_file_buffer)
        actual_file_buffer.seek(0)
        actual_output = actual_file_buffer.read()

        self.assertEqual(actual_output, expected_output)

    def test_patterns_to_file(self):
        data_set = SimulatedDataSetForTests()

        expected_output = ("10.97627008,  14.30378733,  12.05526752\n"
                           "10.89766366,   8.47309599,  12.91788226\n"
                           "8.75174423,  17.83546002,  19.27325521\n"
                           "25.33766075,  41.66900152,  31.15579679\n"
                           "32.72178244,  47.02386553,  12.84144233\n")

        # See: http://stackoverflow.com/a/3945057/1357229
        actual_file_buffer = StringIO()
        data_set._patterns_to_file(actual_file_buffer)
        actual_file_buffer.seek(0)
        actual_output = actual_file_buffer.read()

        self.assertEqual(actual_output, expected_output)

    def test_densities_to_file(self):
        data_set = SimulatedDataSetForTests()

        expected_output = ("7.03125000e-05\n"
                           "6.25000000e-05\n"
                           "6.25000000e-05\n"
                           "7.81250000e-06\n"
                           "7.81250000e-06\n")

        # See: http://stackoverflow.com/a/3945057/1357229
        actual_file_buffer = StringIO()
        data_set._densities_to_file(actual_file_buffer)
        actual_file_buffer.seek(0)
        actual_output = actual_file_buffer.read()

        self.assertEqual(actual_output, expected_output)
