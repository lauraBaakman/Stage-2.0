from io import StringIO
from unittest import TestCase

import numpy as np

from dataset.dataset import DataSet, InvalidDataSetException, _DataSetValidator


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

        self.assertEqual(actual, expected)

    def test_from_file(self):
        self.fail()
        input_file = StringIO("""5 3"""
                              """2 3"""
                              """52.0 45.0 56.0"""
                              """60.0 52.0 41.0"""
                              """37.0 44.0 49.0"""
                              """54.0 56.0 47.0"""
                              """51.0 46.0 47.0"""
                              """7.539699219e-05"""
                              """1.240164051e-05"""
                              """1.227518586e-05"""
                              """7.288289757e-05"""
                              """0.0001832763582""")

        actual = DataSet.from_file(input_file)
        expected = DataSet(
            patterns=np.array([
                [52.0, 45.0, 56.0],
                [60.0, 52.0, 41.0],
                [37.0, 44.0, 49.0],
                [54.0, 56.0, 47.0],
                [51.0, 46.0, 47.0],
            ])
        )

        self.assertAlmostEqual(actual, expected)


class Test_DataSetReader(TestCase):
    def test_read(self):
        input_file = StringIO("""5 3"""
                              """2 3"""
                              """52.0 45.0 56.0"""
                              """60.0 52.0 41.0"""
                              """37.0 44.0 49.0"""
                              """54.0 56.0 47.0"""
                              """51.0 46.0 47.0"""
                              """7.539699219e-05"""
                              """1.240164051e-05"""
                              """1.227518586e-05"""
                              """7.288289757e-05"""
                              """0.0001832763582""")

        actual = DataSet.from_file(input_file)
        expected = DataSet(
            patterns=np.array([
                [52.0, 45.0, 56.0],
                [60.0, 52.0, 41.0],
                [37.0, 44.0, 49.0],
                [54.0, 56.0, 47.0],
                [51.0, 46.0, 47.0],
            ])
        )
        self.assertAlmostEqual(actual, expected)


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