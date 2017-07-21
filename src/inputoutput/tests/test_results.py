from unittest import TestCase
import io

import numpy as np

from inputoutput import DataSet
from inputoutput.results import _ResultsValidator, Results, InvalidResultsException


class TestResults(TestCase):
    def setUp(self):
        super(TestResults, self).setUp()
        self._data_set = DataSet(
                patterns=np.array([
                    [52.0, 45.0, 56.0],
                    [60.0, 52.0, 41.0],
                    [37.0, 44.1, 49.0],
                    [54.0, 56.0, 47.0],
                    [51.0, 46.0, 47.0],
                ]),
                densities=np.array([
                    7.539699219e-05,
                    1.240164051e-05,
                    1.227518586e-05,
                    7.288289757e-05,
                    0.0001832763582,
                ])
            )
        self._results_array = np.array([0.1, 0.2, 0.3, 0.4, 0.51234567891011121314], dtype=np.float64)

    def test_num_results(self):
        results = Results(
            data_set=self._data_set,
            results_array=self._results_array
        )
        actual = results.num_results
        expected = 5
        self.assertEqual(actual, expected)

    def test_to_file(self):
        results = Results(
            data_set=self._data_set,
            results_array=self._results_array
        )
        expected_output = ("0.100000000000000\n"
                           "0.200000000000000\n"
                           "0.300000000000000\n"
                           "0.400000000000000\n"
                           "0.512345678910111\n").encode()
        actual_file_buffer = io.BytesIO()
        results.to_file(actual_file_buffer)
        actual_file_buffer.seek(0)
        actual_output = actual_file_buffer.read()
        self.assertEqual(actual_output, expected_output)

    def test_from_file(self):
        input_file = io.BytesIO(
            """7.539699219e-05\n"""
            """1.240164051e-05\n"""
            """1.227518586e-05\n"""
            """7.288289757e-05\n"""
            """0.0001832763582\n""".encode())

        actual = Results.from_file(input_file)
        expected = Results(
            data_set=None,
            results_array=np.array([
                7.539699219e-05,
                1.240164051e-05,
                1.227518586e-05,
                7.288289757e-05,
                0.0001832763582,
            ])
        )
        self.assertEqual(actual, expected)

    def test__eq_eqal(self):
        one = Results(
            np.array([
                7.539699219e-05,
                1.240164051e-05,
            ])
        )
        two = Results(
            np.array([
                7.539699219e-05,
                1.240164051e-05,
            ])
        )
        self.assertTrue(one == two)

    def test__eq_not_equal(self):
        one = Results(
            np.array([
                7.539699219e-05,
                1.240164051e-05,
            ])
        )
        two = Results(
            np.array([
                7.639699219e-05,
                1.240164051e-05,
            ])
        )
        self.assertFalse(one == two)


class Test_ResultsValidator(TestCase):
    def setUp(self):
        self._data_set = DataSet(
            patterns=np.array([
                [52.0, 45.0, 56.0],
                [60.0, 52.0, 41.0],
                [37.0, 44.0, 49.0],
                [54.0, 56.0, 47.0],
                [51.0, 46.0, 47.0],
            ]),
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
                1.227518586e-05,
                7.288289757e-05,
                0.0001832763582,
            ])
        )

    def test_validate_1(self):
        results_array = np.array([
            0.1, 0.2, 0.3, 0.4, 0.51234567891011121314
        ])
        validator = _ResultsValidator(data_set=self._data_set, results_array=results_array)
        actual = validator.validate()
        self.assertIsNone(actual)

    def test_validate_2(self):
        try:
            results_array = np.array([
                1.0, 2.0, 3.0, 4.0
            ])
            validator = _ResultsValidator(data_set=self._data_set, results_array=results_array)
            validator.validate()
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__one_result_per_pattern_1(self):
        results_array = np.array([
            1.0, 2.0, 3.0, 4.0, 5.1234567891011121314
        ])
        validator = _ResultsValidator(data_set=self._data_set, results_array=results_array)
        actual = validator._one_result_per_pattern()
        self.assertIsNone(actual)

    def test__one_result_per_pattern_2(self):
        try:
            results_array = np.array([
                1.0, 2.0, 3.0, 4.0
            ])
            validator = _ResultsValidator(data_set=self._data_set, results_array=results_array)
            validator._one_result_per_pattern()
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__one_result_per_pattern_3(self):
        try:
            results_array = np.array([
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0
            ])
            validator = _ResultsValidator(data_set=self._data_set, results_array=results_array)
            validator._one_result_per_pattern()
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__dont_check_results_per_pattern_if_bool_is_set(self):
            results_array = np.array([
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
            ])
            validator = _ResultsValidator(data_set=None, results_array=results_array)
            validator.validate()

    def test__results_is_1D_array_1(self):
        results_array = np.array([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0
        ])
        validator = _ResultsValidator(data_set=self._data_set, results_array=results_array)
        actual = validator._results_is_1D_array()
        self.assertIsNone(actual)

    def test__results_is_1D_array_2(self):
        try:
            results_array = np.array([
                [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]
            ])
            validator = _ResultsValidator(data_set=self._data_set, results_array=results_array)
            validator._results_is_1D_array()
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__results_are_densities_0(self):
        results_array = np.array([
            0.0, 0.2, 0.33, 0.444, 0.55, 1.0
        ])
        validator = _ResultsValidator(data_set=self._data_set, results_array=results_array)
        actual = validator._results_is_1D_array()
        self.assertIsNone(actual)

    def test__results_are_densities_1(self):
        results_array = np.array([
            7.539699219e-05,
            1.240164051e-05,
            1.227518586e-05,
            7.288289757e-05,
            0.0001832763582
        ])
        validator = _ResultsValidator(data_set=self._data_set, results_array=results_array)
        actual = validator._results_is_1D_array()
        self.assertIsNone(actual)