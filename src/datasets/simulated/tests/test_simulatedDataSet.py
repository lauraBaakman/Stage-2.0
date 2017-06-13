from unittest import TestCase
from collections import OrderedDict
import io

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
        super(TestSimulatedDataSet, self).setUp()
        self._components = OrderedDict()
        self._components['a'] = component_a
        self._components['b'] = component_b

    def test_components_lengths(self):
        set = SimulatedDataSetForTests()
        actual = set.components_lengths
        expected = [3, 2]
        self.assertEqual(actual, expected)

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

    def test_to_file(self):
        data_set = SimulatedDataSetForTests()

        expected_output = ("5 3\n"
                           "3 2\n"
                           "1.097627007854649506e+01 1.430378732744838999e+01 1.205526752143287794e+01\n"
                           "1.089766365993793684e+01 8.473095986778094613e+00 1.291788226133312278e+01\n"
                           "8.751744225253849763e+00 1.783546001564159411e+01 1.927325521002058650e+01\n"
                           "2.533766075303110910e+01 4.166900152330658358e+01 3.115579679011617742e+01\n"
                           "3.272178244375729150e+01 4.702386553170644135e+01 1.284144232791547680e+01\n"
                           "7.031250000000002179e-05\n"
                           "6.250000000000001485e-05\n"
                           "6.250000000000001485e-05\n"
                           "7.812500000000001857e-06\n"
                           "7.812500000000001857e-06\n").encode()

        # See: http://stackoverflow.com/a/3945057/1357229
        actual_file_buffer = io.BytesIO()
        data_set.to_file(actual_file_buffer)
        actual_file_buffer.seek(0)
        actual_output = actual_file_buffer.read()

        self.assertEqual(actual_output, expected_output)

    def test__header_to_file(self):
        data_set = SimulatedDataSetForTests()

        expected_output = ("5 3\n"
                           "3 2\n").encode()

        actual_file_buffer = io.BytesIO()
        data_set._header_to_file(actual_file_buffer)
        actual_file_buffer.seek(0)
        actual_output = actual_file_buffer.read()

        self.assertEqual(actual_output, expected_output)

    def test_patterns_to_file(self):
        data_set = SimulatedDataSetForTests()

        expected_output = ("1.097627007854649506e+01 1.430378732744838999e+01 1.205526752143287794e+01\n"
                           "1.089766365993793684e+01 8.473095986778094613e+00 1.291788226133312278e+01\n"
                           "8.751744225253849763e+00 1.783546001564159411e+01 1.927325521002058650e+01\n"
                           "2.533766075303110910e+01 4.166900152330658358e+01 3.115579679011617742e+01\n"
                           "3.272178244375729150e+01 4.702386553170644135e+01 1.284144232791547680e+01\n").encode()

        # See: http://stackoverflow.com/a/3945057/1357229
        actual_file_buffer = io.BytesIO()
        data_set._patterns_to_file(actual_file_buffer)
        actual_file_buffer.seek(0)
        actual_output = actual_file_buffer.read()

        self.assertEqual(actual_output, expected_output)

    def test_densities_to_file(self):
        data_set = SimulatedDataSetForTests()


        expected_output = ("7.031250000000002179e-05\n"
                           "6.250000000000001485e-05\n"
                           "6.250000000000001485e-05\n"
                           "7.812500000000001857e-06\n"
                           "7.812500000000001857e-06\n").encode()

        # See: http://stackoverflow.com/a/3945057/1357229
        actual_file_buffer = io.BytesIO()
        data_set._densities_to_file(actual_file_buffer)
        actual_file_buffer.seek(0)
        actual_output = actual_file_buffer.read()

        self.assertEqual(actual_output, expected_output)

    def test_reading_and_writing(self):
        expected_data_set = SimulatedDataSetForTests()

        file_buffer = io.BytesIO()
        expected_data_set.to_file(file_buffer)
        file_buffer.seek(0)

        actual_data_set = SimulatedDataSetForTests.from_file(file_buffer)
        self.assertEqual(expected_data_set, actual_data_set)
