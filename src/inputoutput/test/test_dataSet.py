from io import BytesIO
from unittest import TestCase

import numpy as np

from inputoutput.dataset import DataSet, InvalidDataSetException, _DataSetValidator, _DataSetReader


class TestDataSet(TestCase):
    def setUp(self):
        super(TestDataSet, self).setUp()

    def test_has_densities_true(self):
        data_set = DataSet(
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
        actual = data_set.has_densities
        self.assertTrue(actual)

    def test_has_densities_false(self):
        data_set = DataSet(
            patterns=np.array([
                [52.0, 45.0, 56.0],
                [60.0, 52.0, 41.0],
                [37.0, 44.0, 49.0],
                [54.0, 56.0, 47.0],
                [51.0, 46.0, 47.0],
            ])
        )
        actual = data_set.has_densities
        self.assertFalse(actual)

    def test_validate_complete_set(self):
        DataSet(
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

    def test_validate_set_without_densities(self):
        DataSet(
            patterns=np.array([
                [52.0, 45.0, 56.0],
                [60.0, 52.0, 41.0],
                [37.0, 44.0, 49.0],
                [54.0, 56.0, 47.0],
                [51.0, 46.0, 47.0],
            ])
        )

    def test_to_file_no_densities(self):
        data_set = DataSet(
            patterns=np.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [0.5, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ])
        )
        # See: http://stackoverflow.com/a/3945057/1357229
        actual_file_buffer = BytesIO()
        data_set.to_file(actual_file_buffer)
        actual_file_buffer.seek(0)
        actual_data_set = DataSet.from_file(actual_file_buffer)
        self.assertEqual(actual_data_set, data_set)

    def test_to_file_with_densities(self):
        data_set = DataSet(
            patterns=np.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [0.5, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ]),
            densities=np.array([
                0.000007031250000000002179,
                0.000006250000000000001485,
                0.000006250000000000001485,
                0.0000007812500000000001857,
                0.0000007812500000000001857
            ])
        )
        # See: http://stackoverflow.com/a/3945057/1357229
        actual_file_buffer = BytesIO()
        data_set.to_file(actual_file_buffer)
        actual_file_buffer.seek(0)
        actual_data_set = DataSet.from_file(actual_file_buffer)
        self.assertEqual(actual_data_set, data_set)

    def test_num_patterns(self):
        data_set = DataSet(
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
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
                1.227518586e-05,
                7.288289757e-05,
                0.0001832763582,
            ])
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
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
                1.227518586e-05,
                7.288289757e-05,
                0.0001832763582,
            ])
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
            ]),
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
                1.227518586e-05,
                7.288289757e-05,
                0.0001832763582,
            ])
        )
        self.assertEqual(actual, expected)

    def test__eq_both_eqal(self):
        one = DataSet(
            patterns=np.array([
                [52.0, 45.0, 56.0],
                [60.0, 52.0, 41.0],
            ]),
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
            ])
        )
        two = DataSet(
            patterns=np.array([
                [52.0, 45.0, 56.0],
                [60.0, 52.0, 41.0],
            ]),
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
            ])
        )
        self.assertTrue(one == two)

    def test__eq_patterns_not_equal(self):
        one = DataSet(
            patterns=np.array([
                [52.0, 45.0, 56.0],
                [60.0, 52.0, 41.0],
            ]),
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
            ])
        )
        two = DataSet(
            patterns=np.array([
                [52.0, 45.0, 56.0],
                [60.0, 55.0, 41.0],
            ]),
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
            ])
        )
        self.assertFalse(one == two)

    def test__eq_densities_not_equal(self):
        one = DataSet(
            patterns=np.array([
                [52.0, 45.0, 56.0],
                [60.0, 52.0, 41.0],
            ]),
            densities=np.array([
                7.539699219e-05,
                1.250164051e-05,
            ])
        )
        two = DataSet(
            patterns=np.array([
                [52.0, 45.0, 56.0],
                [60.0, 52.0, 41.0],
            ]),
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
            ])
        )
        self.assertFalse(one == two)


class Test_DataSetReader(TestCase):
    def setUp(self):
        super(Test_DataSetReader, self).setUpClass()
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

    def test_read_with_densities(self):
        actual = _DataSetReader(self._input_file).read()
        expected = DataSet(
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
        self.assertEqual(actual, expected)

    def test_read_no_densities(self):
        input_file = BytesIO(
            """5 3\n"""
            """2 3\n"""
            """52.0 45.0 56.0\n"""
            """60.0 52.0 41.0\n"""
            """37.0 44.1 49.0\n"""
            """54.0 56.0 47.0\n"""
            """51.0 46.0 47.0\n""".encode()
        )
        actual = _DataSetReader(input_file).read()
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
        actual = reader._read_densities()
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
        densities = np.array([
            7.539699219e-05,
            1.240164051e-05,
            1.227518586e-05,
            7.288289757e-05,
            0.0001832763582,
        ])
        validator = _DataSetValidator(patterns=patterns, densities=densities)
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
            densities = np.array([
                7.539699219e-05,
                1.240164051e-05,
                1.227518586e-05,
                7.288289757e-05,
                0.0001832763582,
            ])
            validator = _DataSetValidator(patterns=patterns, densities=densities)
            validator.validate()
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
        densities = np.array([
            7.539699219e-05,
            1.240164051e-05,
            1.227518586e-05,
            7.288289757e-05,
            0.0001832763582,
        ])
        validator = _DataSetValidator(patterns=patterns, densities=densities)
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
            densities = np.array([
                7.539699219e-05,
                1.240164051e-05,
                1.227518586e-05,
                7.288289757e-05,
                0.0001832763582,
            ])
            validator = _DataSetValidator(patterns=patterns, densities=densities)
            validator._patterns_is_2D_array()
        except InvalidDataSetException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__densities_is_1D_array_1(self):
        patterns = np.array([
            [52.0, 45.0, 56.0],
            [60.0, 52.0, 41.0],
            [37.0, 44.0, 49.0],
            [54.0, 56.0, 47.0],
            [51.0, 46.0, 47.0],
        ])
        densities = np.array([
            7.539699219e-05,
            1.240164051e-05,
            1.227518586e-05,
            7.288289757e-05,
            0.0001832763582,
        ])
        validator = _DataSetValidator(patterns=patterns, densities=densities)
        actual = validator._densities_is_1D_array()
        self.assertIsNone(actual)

    def test__densities_is_1D_array_2(self):
        patterns = np.array([
            [[52.0, 45.0, 56.0],
             [60.0, 52.0, 41.0],
             [37.0, 44.0, 49.0]],
            [[54.0, 56.0, 47.0],
             [51.0, 46.0, 47.0]],
        ])
        densities = np.array([
            [7.539699219e-05, 1.240164051e-05, ],
            [7.288289757e-05, 0.0001832763582],
        ])

        try:
            validator = _DataSetValidator(patterns=patterns, densities=densities)
            validator._densities_is_1D_array()
        except InvalidDataSetException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__num_labels_equals_num_patterns_0(self):
        patterns = np.array([
            [52.0, 45.0, 56.0],
            [60.0, 52.0, 41.0],
            [37.0, 44.0, 49.0],
            [54.0, 56.0, 47.0],
            [51.0, 46.0, 47.0],
        ])
        densities = np.array([
            7.539699219e-05,
            1.240164051e-05,
            1.227518586e-05,
            7.288289757e-05,
            0.0001832763582,
        ])
        validator = _DataSetValidator(patterns=patterns, densities=densities)
        actual = validator._num_labels_equals_num_patterns()
        self.assertIsNone(actual)

    def test__num_labels_equals_num_patterns_1(self):
        patterns = np.array([
            [52.0, 45.0, 56.0],
            [60.0, 52.0, 41.0],
            [37.0, 44.0, 49.0],
            [54.0, 56.0, 47.0],
            [51.0, 46.0, 47.0],
        ])
        densities = np.array([
            7.539699219e-05,
            1.240164051e-05,
            1.227518586e-05,
        ])

        try:
            validator = _DataSetValidator(patterns=patterns, densities=densities)
            validator._num_labels_equals_num_patterns()
        except InvalidDataSetException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__num_labels_equals_num_patterns_2(self):
        patterns = np.array([
            [52.0, 45.0, 56.0],
            [60.0, 52.0, 41.0],
            [37.0, 44.0, 49.0],
            [54.0, 56.0, 47.0],
            [51.0, 46.0, 47.0],
        ])
        densities = np.array([
            7.539699219e-05,
            1.240164051e-05,
            1.227518586e-05,
            7.288289757e-05,
            0.0001832763582,
            7.288289757e-05,
            0.0001832763582,
        ])

        try:
            validator = _DataSetValidator(patterns=patterns, densities=densities)
            validator._num_labels_equals_num_patterns()
        except InvalidDataSetException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__densities_is_probability_densities_0(self):
        patterns = np.array([
            [52.0, 45.0, 56.0],
            [60.0, 52.0, 41.0],
            [37.0, 44.0, 49.0],
            [54.0, 56.0, 47.0],
            [51.0, 46.0, 47.0],
        ])
        densities = np.array([
            7.539699219e-05,
            0.0,
            1.227518586e-05,
            7.288289757e-05,
            0.0001832763582,
            7.288289757e-05,
            1.0,
        ])
        validator = _DataSetValidator(patterns=patterns, densities=densities)
        actual = validator._densities_is_probability_densities()
        self.assertIsNone(actual)

    def test__densities_is_probability_densities_1(self):
        patterns = np.array([
            [52.0, 45.0, 56.0],
            [60.0, 52.0, 41.0],
            [37.0, 44.0, 49.0],
            [54.0, 56.0, 47.0],
            [51.0, 46.0, 47.0],
        ])
        densities = np.array([
            7.539699219e-05,
            1.240164051e-05,
            1.227518586e-05,
            7.288289757e-05,
            0.0001832763582,
            7.288289757,
            0.0001832763582,
        ])

        try:
            validator = _DataSetValidator(patterns=patterns, densities=densities)
            validator._densities_is_probability_densities()
        except InvalidDataSetException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__densities_is_probability_densities_2(self):
        patterns = np.array([
            [52.0, 45.0, 56.0],
            [60.0, 52.0, 41.0],
            [37.0, 44.0, 49.0],
            [54.0, 56.0, 47.0],
            [51.0, 46.0, 47.0],
        ])
        densities = np.array([
            7.539699219e-05,
            1.240164051e-05,
            1.227518586e-05,
            7.288289757e-05,
            0.0001832763582,
            7.288289757e-05,
            -0.0001832763582,
        ])

        try:
            validator = _DataSetValidator(patterns=patterns, densities=densities)
            validator._densities_is_probability_densities()
        except InvalidDataSetException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')
