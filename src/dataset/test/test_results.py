from unittest import TestCase

import numpy as np

from dataset import DataSet
from dataset.results import _ResultsValidator, Results, InvalidResultsException


class TestResults(TestCase):
    def test_num_results(self):
        results = Results(
            data_set=DataSet(
                patterns=np.array([
                    [52.0, 45.0, 56.0],
                    [60.0, 52.0, 41.0],
                    [37.0, 44.1, 49.0],
                    [54.0, 56.0, 47.0],
                    [51.0, 46.0, 47.0],
                ])
            ),
            results_array=np.array([1.0, 2.0, 3.0, 4.0, 5.1234567891011121314])
        )
        actual = results.num_results
        expected = 5
        self.assertEqual(actual, expected)

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
        )

    def test_validate_1(self):
        results_array = np.array([
            1.0, 2.0, 3.0, 4.0, 5.1234567891011121314
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
            actual = validator.validate()
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
            actual = validator._one_result_per_pattern()
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
            actual = validator._one_result_per_pattern()
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

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
            actual = validator._results_is_1D_array()
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')
