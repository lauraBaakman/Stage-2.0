from io import BytesIO
from unittest import TestCase

import numpy as np

from inputoutput.dataset import DataSet, InvalidDataSetException, _DataSetValidator, _DataSetReader


class TestDataSet(TestCase):
    def setUp(self):
        super().setUp()
        self._data_set = DataSet(
            patterns=np.array([
                [52.0, 45.0, 56.0],
                [60.0, 52.0, 41.0],
                [37.0, 44.0, 49.0],
                [54.0, 56.0, 47.0],
                [51.0, 46.0, 47.0],
            ]),
        )

    def test_num_patterns(self):
        data_set = DataSet(
            patterns=np.array([
                [52.0, 45.0, 56.0],
                [60.0, 52.0, 41.0],
                [37.0, 44.0, 49.0],
                [54.0, 56.0, 47.0],
                [51.0, 46.0, 47.0],
            ]),
        )
        data_set.results = np.array([
                1.0, 2.0, 3.0, 4.0, 5.1234567891011121314
            ])
        actual = data_set.num_patterns
        expected = 5

        self.assertEqual(actual, expected)

    def test_pattern_dimension(self):
        data_set = DataSet(
            patterns=np.array([
                [52.0, 45.0, 56.0],
                [60.0, 52.0, 41.0],
                [37.0, 44.0, 49.0],
                [54.0, 56.0, 47.0],
                [51.0, 46.0, 47.0],
            ]),
        )
        actual = data_set.dimension
        expected = 3

        self.assertEqual(actual, expected)

    def test_patterns(self):
        data_set = DataSet(
            patterns=np.array([
                [52.0, 45.0, 56.0],
                [60.0, 52.0, 41.0],
                [37.0, 44.0, 49.0],
                [54.0, 56.0, 47.0],
                [51.0, 46.0, 47.0],
            ]),
        )
        actual = data_set.patterns
        expected = np.array([
            [52.0, 45.0, 56.0],
            [60.0, 52.0, 41.0],
            [37.0, 44.0, 49.0],
            [54.0, 56.0, 47.0],
            [51.0, 46.0, 47.0],
        ])

        np.testing.assert_array_equal(actual, expected)

    def test_from_file(self):
        input_file = BytesIO("""5 3\n"""
                             """2 3\n"""
                             """52.0 45.0 56.0\n"""
                             """60.0 52.0 41.0\n"""
                             """37.0 44.1 49.0\n"""
                             """54.0 56.0 47.0\n"""
                             """51.0 46.0 47.0\n"""
                             """7.539699219e-05\n"""
                             """1.240164051e-05\n"""
                             """1.227518586e-05\n"""
                             """7.288289757e-05\n"""
                             """0.0001832763582\n""".encode())

        actual = DataSet.from_file(input_file)
        expected = DataSet(
            patterns=np.array([
                [52.0, 45.0, 56.0],
                [60.0, 52.0, 41.0],
                [37.0, 44.1, 49.0],
                [54.0, 56.0, 47.0],
                [51.0, 46.0, 47.0],
            ])
        )

        self.assertEqual(actual, expected)


class Test_DataSetReader(TestCase):
    def setUp(self):
        super().setUpClass()
        self._input_file = BytesIO("""5 3\n"""
                                   """2 3\n"""
                                   """52.0 45.0 56.0\n"""
                                   """60.0 52.0 41.0\n"""
                                   """37.0 44.1 49.0\n"""
                                   """54.0 56.0 47.0\n"""
                                   """51.0 46.0 47.0\n"""
                                   """7.539699219e-05\n"""
                                   """1.240164051e-05\n"""
                                   """1.227518586e-05\n"""
                                   """7.288289757e-05\n"""
                                   """0.0001832763582\n""".encode())

    def test_read(self):
        actual = _DataSetReader(self._input_file).read()
        expected = DataSet(
            patterns=np.array([
                [52.0, 45.0, 56.0],
                [60.0, 52.0, 41.0],
                [37.0, 44.1, 49.0],
                [54.0, 56.0, 47.0],
                [51.0, 46.0, 47.0],
            ])
        )
        self.assertEqual(actual, expected)

    def test__read_pattern_count(self):
        actual = _DataSetReader(self._input_file)._read_pattern_count()
        expected = 5
        self.assertEqual(actual, expected)

    def test__read_patterns(self):
        reader = _DataSetReader(self._input_file)
        reader._num_patterns = 5
        actual = reader._read_patterns()
        expected = np.array([
            [52.0, 45.0, 56.0],
            [60.0, 52.0, 41.0],
            [37.0, 44.1, 49.0],
            [54.0, 56.0, 47.0],
            [51.0, 46.0, 47.0],
        ])
        np.testing.assert_array_equal(actual, expected)

    def test__read_labels(self):
        reader = _DataSetReader(self._input_file)
        reader._num_patterns = 5
        actual = reader._read_labels()
        expected = np.array([
            7.539699219e-05,
            1.240164051e-05,
            1.227518586e-05,
            7.288289757e-05,
            0.0001832763582
        ])
        np.testing.assert_array_equal(actual, expected)


class Test_DataSetValidator(TestCase):
    def test_validate_1(self):
        patterns = np.array([
            [52.0, 45.0, 56.0],
            [60.0, 52.0, 41.0],
            [37.0, 44.0, 49.0],
            [54.0, 56.0, 47.0],
            [51.0, 46.0, 47.0],
        ])
        validator = _DataSetValidator(patterns=patterns)
        actual = validator.validate()
        self.assertIsNone(actual)

    def test_validate_2(self):
        try:
            patterns = np.array([
                [[52.0, 45.0, 56.0],
                 [60.0, 52.0, 41.0],
                 [37.0, 44.0, 49.0]],
                [[54.0, 56.0, 47.0],
                 [51.0, 46.0, 47.0]],
            ])
            validator = _DataSetValidator(patterns=patterns)
            actual = validator.validate()
        except InvalidDataSetException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__patterns_is_2D_array_1(self):
        patterns = np.array([
            [52.0, 45.0, 56.0],
            [60.0, 52.0, 41.0],
            [37.0, 44.0, 49.0],
            [54.0, 56.0, 47.0],
            [51.0, 46.0, 47.0],
        ])
        validator = _DataSetValidator(patterns=patterns)
        actual = validator._patterns_is_2D_array()
        self.assertIsNone(actual)

    def test__patterns_is_2D_array_2(self):
        try:
            patterns = np.array([
                [[52.0, 45.0, 56.0],
                 [60.0, 52.0, 41.0],
                 [37.0, 44.0, 49.0]],
                [[54.0, 56.0, 47.0],
                 [51.0, 46.0, 47.0]],
            ])
            validator = _DataSetValidator(patterns=patterns)
            actual = validator._patterns_is_2D_array()
        except InvalidDataSetException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')