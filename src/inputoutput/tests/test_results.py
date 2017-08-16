from unittest import TestCase, expectedFailure
import io
import shutil
import tempfile
import warnings
import exceptions

import numpy as np
from unipath import Path

from inputoutput import DataSet
from inputoutput.results import _DensitiesValidator, _XisValidator, Results, InvalidResultsException


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
        self._densities = np.array([0.1, 0.2, 0.3, 0.4, 0.51234567891011121314], dtype=np.float64)
        self.test_dir = Path(tempfile.mkdtemp())
        warnings.simplefilter("always")

    def tearDown(self):
        super(TestResults, self).tearDown()
        shutil.rmtree(self.test_dir)

    def test_constructor_without_densities(self):
        expected_size = 4
        results = Results(expected_size=expected_size)
        actual = results.densities
        expected = np.empty(expected_size)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.flags, expected.flags)
        self.assertEqual(actual.dtype, expected.dtype)

    def test_constructor_with_densities(self):
        results = Results(
            densities=self._densities
        )
        actual = results.densities
        expected = self._densities
        np.testing.assert_array_almost_equal(actual, expected)

    def test_constructor_without_densities_without_expected_size(self):
        try:
            Results()
        except TypeError:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_add_xis(self):
        actual = Results(densities=self._densities)
        xis = np.random.rand(100, 3)
        actual.xis = xis

        np.testing.assert_array_almost_equal(actual.xis, xis)

    def test_num_results(self):
        results = Results(
            data_set=self._data_set,
            densities=self._densities
        )
        actual = results.num_results
        expected = 5
        self.assertEqual(actual, expected)

    def test_has_num_used_patterns_true(self):
        results = Results(
            data_set=None,
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
                1.227518586e-05,
                7.288289757e-05,
                0.0001832763582,
            ]),
            num_used_patterns=np.array([1, 2, 3, 4, 5])
        )
        self.assertTrue(results.has_num_used_patterns)

    def test_has_num_used_patterns_true_only_nans(self):
        results = Results(
            data_set=None,
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
                1.227518586e-05,
                7.288289757e-05,
                0.0001832763582,
            ]),
            num_used_patterns=np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        )
        self.assertTrue(results.has_num_used_patterns)

    def test_has_num_used_patterns_false(self):
        results = Results(
            data_set=None,
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
                1.227518586e-05,
                7.288289757e-05,
                0.0001832763582,
            ])
        )
        self.assertFalse(results.has_num_used_patterns)

    def test_scaling_factors(self):
        scaling_factors = np.random.rand(100)
        results = Results(
            data_set=None,
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
                1.227518586e-05,
                7.288289757e-05,
                0.0001832763582,
            ]),
            scaling_factors=scaling_factors
        )
        np.testing.assert_array_equal(scaling_factors, results.scaling_factors)

    def test_num_used_patterns(self):
        array = np.array([
                7.539699219e-05,
                1.240164051e-05,
            ])
        results = Results(
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
            ]),
            num_used_patterns=array
        )
        np.testing.assert_array_almost_equal(
            results.num_used_patterns,
            array
        )

    def test_to_file(self):
        results = Results(
            data_set=self._data_set,
            densities=self._densities
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

    def test_eigen_vectors_to_file(self):
        xis = np.array([
            [52.0, 45.0, 56.0],
            [60.0, 52.0, 41.0]])
        results = Results(
            data_set=self._data_set,
            densities=self._densities,
            xis=xis,
            eigen_vectors=np.array([
                [[0, 1, 2],
                 [3, 4, 5],
                 [6, 7, 8]],
                [
                 [9, 8, 7],
                 [6, 5, 4],
                 [3, 2, 1]
                ]
            ])
        )
        expected_output = (
            "# xi_x xi_y xi_z "
            "eigen_vector_1_x eigen_vector_1_y eigen_vector_1_z "
            "eigen_vector_2_x eigen_vector_2_y eigen_vector_2_z "
            "eigen_vector_3_x eigen_vector_3_y eigen_vector_3_z\n"
            "52.000000000000000 45.000000000000000 56.000000000000000 "
            "0.000000000000000 1.000000000000000 2.000000000000000 "
            "3.000000000000000 4.000000000000000 5.000000000000000 "
            "6.000000000000000 7.000000000000000 8.000000000000000\n"
            "60.000000000000000 52.000000000000000 41.000000000000000 "
            "9.000000000000000 8.000000000000000 7.000000000000000 "
            "6.000000000000000 5.000000000000000 4.000000000000000 "
            "3.000000000000000 2.000000000000000 1.000000000000000\n"
        ).encode()
        x_file_buffer = io.BytesIO()
        xi_file_buffer = io.BytesIO()
        results.to_file(x_file_buffer, xi_file_buffer)
        xi_file_buffer.seek(0)
        actual_output = xi_file_buffer.read()
        self.assertEqual(actual_output, expected_output)

    def test_raises_warning_if_writting_le_3d_xis_data(self):
        num_xis = 20
        num_xs = 5
        dimension = 2
        expected = Results(
            densities=np.random.rand(num_xs),
            num_used_patterns=np.round(np.random.rand(num_xs)),
            xis=np.random.rand(num_xis, dimension),
            eigen_values=np.random.rand(num_xis, dimension),
        )
        x_out_path = self.test_dir.child('x_out.txt')
        xi_out_path = self.test_dir.child('xi_out.txt')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with open(x_out_path, 'wb') as x_handle, open(xi_out_path, 'wb') as xi_handle:
                expected.to_file(x_handle, xi_handle)
            if any([warning.category is exceptions.UserWarning for warning in w]):
                self.assertTrue(True)
            else:
                self.fail('Expected warning not thrown.')

    def test_raises_warning_if_writting_ge_3d_xis_data(self):
        num_xis = 20
        num_xs = 5
        dimension = 5
        expected = Results(
            densities=np.random.rand(num_xs),
            num_used_patterns=np.round(np.random.rand(num_xs)),
            xis=np.random.rand(num_xis, dimension),
            eigen_values=np.random.rand(num_xis, dimension),
        )
        x_out_path = self.test_dir.child('x_out.txt')
        xi_out_path = self.test_dir.child('xi_out.txt')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with open(x_out_path, 'wb') as x_handle, open(xi_out_path, 'wb') as xi_handle:
                expected.to_file(x_handle, xi_handle)
            if any([warning.category is exceptions.UserWarning for warning in w]):
                self.assertTrue(True)
            else:
                self.fail('Expected warning not thrown.')

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
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
                1.227518586e-05,
                7.288289757e-05,
                0.0001832763582,
            ])
        )
        self.assertEqual(actual, expected)

    def test_from_file_to_file_with_temp_file(self):
        expected = Results(
            data_set=None,
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
                1.227518586e-05,
                7.288289757e-05,
                0.0001832763582,
            ])
        )
        out_path = self.test_dir.child('temp.txt')

        with open(out_path, 'w') as out_handle:
            expected.to_file(out_handle)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            actual = Results.from_file(out_path)
            if len(w):
                self.fail('Some warning was triggered')
        self.assertEqual(actual, expected)

    def test_from_file_to_files_with_temp_file_and_xis_file_all_fields(self):
        num_xis = 20
        num_xs = 5
        dimension = 3
        expected = Results(
            densities=np.random.rand(num_xs),
            num_used_patterns=np.round(np.random.rand(num_xs)),
            xis=np.random.rand(num_xis, dimension),
            eigen_values=np.random.rand(num_xis, dimension),
            eigen_vectors=np.random.rand(num_xis, dimension, dimension),
            scaling_factors=np.random.rand(num_xis)
        )
        x_out_path = self.test_dir.child('x_out.txt')
        xi_out_path = self.test_dir.child('xi_out.txt')

        with open(x_out_path, 'w') as x_handle, open(xi_out_path, 'w') as xi_handle:
            expected.to_file(x_handle, xi_handle)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            actual = Results.from_file(x_out_path, xi_out_path)
            if len(w):
                self.fail('Some warning was triggered')
        self.assertEqual(actual, expected)

    def test_from_file_to_files_with_temp_file_and_xis_file_only_xis(self):
        num_xis = 20
        num_xs = 5
        dimension = 3
        expected = Results(
            densities=np.random.rand(num_xs),
            num_used_patterns=np.round(np.random.rand(num_xs)),
            xis=np.random.rand(num_xis, dimension),
        )
        x_out_path = self.test_dir.child('x_out.txt')
        xi_out_path = self.test_dir.child('xi_out.txt')

        with open(x_out_path, 'wb') as x_handle, open(xi_out_path, 'wb') as xi_handle:
            expected.to_file(x_handle, xi_handle)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            actual = Results.from_file(x_out_path, xi_out_path)
            if len(w) and any([warning.category is not exceptions.DeprecationWarning for warning in w]):
                self.fail('Some warning was triggered')
        self.assertEqual(actual, expected)

    def test_from_file_to_files_with_temp_file_and_xis_file_only_xis_eigen_values(self):
        num_xis = 20
        num_xs = 5
        dimension = 3
        expected = Results(
            densities=np.random.rand(num_xs),
            num_used_patterns=np.round(np.random.rand(num_xs)),
            xis=np.random.rand(num_xis, dimension),
            eigen_values=np.random.rand(num_xis, dimension),
        )
        x_out_path = self.test_dir.child('x_out.txt')
        xi_out_path = self.test_dir.child('xi_out.txt')

        with open(x_out_path, 'wb') as x_handle, open(xi_out_path, 'wb') as xi_handle:
            expected.to_file(x_handle, xi_handle)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            actual = Results.from_file(x_out_path, xi_out_path)
            if len(w):
                self.fail('Some warning was triggered')
        self.assertEqual(actual, expected)

    def test_from_file_to_files_with_temp_file_and_xis_file_scaling_factor_absent(self):
        num_xis = 4
        num_xs = 2
        dimension = 3
        expected = Results(
            densities=np.random.rand(num_xs),
            num_used_patterns=np.round(np.random.rand(num_xs)),
            xis=np.random.rand(num_xis, dimension),
            eigen_values=np.random.rand(num_xis, dimension),
            eigen_vectors=np.random.rand(num_xis, dimension, dimension),
        )
        x_out_path = self.test_dir.child('x_out.txt')
        xi_out_path = self.test_dir.child('xi_out.txt')

        with open(x_out_path, 'w') as x_handle, open(xi_out_path, 'w') as xi_handle:
            expected.to_file(x_handle, xi_handle)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            actual = Results.from_file(x_out_path, xi_out_path)
            if len(w) and any([warning.category is not exceptions.DeprecationWarning for warning in w]):
                self.fail('Some warning was triggered')
        self.assertEqual(actual, expected)

    def test_from_file_to_files_with_temp_file_and_xis_file_no_eigen_properties(self):
        num_xis = 20
        num_xs = 5
        dimension = 3
        expected = Results(
            densities=np.random.rand(num_xs),
            num_used_patterns=np.round(np.random.rand(num_xs)),
            xis=np.random.rand(num_xis, dimension),
            scaling_factors=np.random.rand(num_xis)
        )
        x_out_path = self.test_dir.child('x_out.txt')
        xi_out_path = self.test_dir.child('xi_out.txt')

        with open(x_out_path, 'w') as x_handle, open(xi_out_path, 'w') as xi_handle:
            expected.to_file(x_handle, xi_handle)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            actual = Results.from_file(x_out_path, xi_out_path)
            if len(w):
                self.fail('Some warning was triggered')
        self.assertEqual(actual, expected)

    def test_from_file_to_file_with_num_used_patterns(self):
        expected = Results(
            data_set=None,
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
                1.227518586e-05,
                7.288289757e-05,
                0.0001832763582,
            ]),
            num_used_patterns=np.array([1, 2, 3, np.nan, 5])
        )
        out_path = self.test_dir.child('temp.txt')

        with open(out_path, 'w') as out_handle:
            expected.to_file(out_handle)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            actual = Results.from_file(out_path)
            if len(w):
                self.fail('Some warning was triggered: {}'.format(w[0].message))
        self.assertEqual(actual, expected)

    def test_is_incremental_true(self):
        results = Results(expected_size=3)
        self.assertTrue(results.is_incremental)

    def test_is_incremental_false(self):
        results = Results(densities=self._densities)
        self.assertFalse(results.is_incremental)

    def test_add_result_only_density(self):
        actual = Results(expected_size=3)
        actual.add_result(density=0.5)
        actual.add_result(density=0.3)
        actual.add_result(density=0.2)

        expected = Results(
            densities=np.array([0.5, 0.3, 0.2]),
            num_used_patterns=np.array([np.nan, np.nan, np.nan], dtype=np.float)
        )
        self.assertEqual(actual, expected)

    def test_add_result_only_density_invalid_density(self):
        actual = Results(expected_size=3)
        with warnings.catch_warnings(record=True) as w:
            actual.add_result(density=0.5)
            actual.add_result(density=3.0)
            actual.add_result(density=0.2)
            if w and issubclass(w[0].category, UserWarning):
                pass
            else:
                self.fail('Expected warning not thrown')
            expected = Results(
                densities=np.array([0.5, 3.0, 0.2]),
                num_used_patterns=np.array([np.nan, np.nan, np.nan])
            )
            self.assertEqual(actual, expected)

    def test_add_results_to_result_initialize_with_densities(self):
        try:
            results = Results(densities=self._densities)
            results.add_result(0.5)
        except TypeError:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_add_result_density_and_count_1(self):
        actual = Results(expected_size=3)
        actual.add_result(density=0.5, num_used_patterns=2)
        actual.add_result(density=0.3, num_used_patterns=4)
        actual.add_result(density=0.2, num_used_patterns=3)

        expected = Results(
            densities=np.array([0.5, 0.3, 0.2]),
            # needs to be float, so that we can use nan for the unknown values
            num_used_patterns=np.array([2, 4, 3], dtype=np.float)
        )
        self.assertEqual(actual, expected)
        self.assertEqual(actual.num_used_patterns.dtype,
                         expected.num_used_patterns.dtype)

    def test_add_result_density_and_count_2(self):
        actual = Results(expected_size=3)
        actual.add_result(density=0.5, num_used_patterns=2)
        actual.add_result(density=0.3)
        actual.add_result(density=0.2, num_used_patterns=3)

        expected = Results(
            densities=np.array([0.5, 0.3, 0.2]),
            num_used_patterns=np.array([2, np.nan, 3], dtype=np.float)
        )
        self.assertEqual(actual, expected)
        self.assertEqual(actual.num_used_patterns.dtype,
                         expected.num_used_patterns.dtype)

    def test_result_fault_key(self):
        try:
            actual = Results(expected_size=1)
            actual.add_result(density=0.5, wrong_key_name=2)
        except KeyError:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

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

    def test__eq_equal_with_count(self):
        one = Results(
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
            ]),
            num_used_patterns=np.array([2, 3])
        )
        two = Results(
            np.array([
                7.539699219e-05,
                1.240164051e-05,
            ]),
            num_used_patterns=np.array([2, 3])
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

    def test__eq_with_nan(self):
        one = Results(
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
            ]),
            num_used_patterns=np.array([2, np.nan])
        )
        two = Results(
            np.array([
                7.539699219e-05,
                1.240164051e-05,
            ]),
            num_used_patterns=np.array([2, np.nan])
        )
        self.assertTrue(one == two)

    def test__eq_with_nan_2(self):
        one = Results(
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
            ]),
            num_used_patterns=np.array([np.nan, 2])
        )
        two = Results(
            np.array([
                7.539699219e-05,
                1.240164051e-05,
            ]),
            num_used_patterns=np.array([2, np.nan])
        )
        self.assertFalse(one == two)

    def test__eq_with_none(self):
        one = Results(
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
            ])
        )
        two = Results(
            np.array([
                7.539699219e-05,
                1.240164051e-05,
            ]),
            num_used_patterns=np.array([2, 3])
        )
        self.assertFalse(one == two)

    def test__eq_not_equal_with_count(self):
        one = Results(
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
            ]),
            num_used_patterns=np.array([2, 3])
        )
        two = Results(
            np.array([
                7.539699219e-05,
                1.240164051e-05,
            ]),
            num_used_patterns=np.array([4, 7])
        )
        self.assertFalse(one == two)

    def test__eq_with_differing_properties(self):
        one = Results(
            densities=np.array([
                7.539699219e-05,
                1.240164051e-05,
            ])
        )
        two = Results(
            np.array([
                7.539699219e-05,
                1.240164051e-05,
            ]),
            num_used_patterns=np.array([4, 7])
        )
        self.assertFalse(one == two)

    def test_eq_with_all_equal(self):
        num_xs = 10
        num_xis = 200
        dimension = 3
        one = Results(
            densities=np.random.rand(num_xs),
            num_used_patterns=np.random.rand(num_xs),
            xis=np.random.rand(num_xis, dimension),
            eigen_values=np.random.rand(num_xis, dimension),
            eigen_vectors=np.random.rand(num_xis, dimension, dimension),
            scaling_factors=np.random.rand(num_xis)
        )
        two = Results(
            densities=one.densities,
            num_used_patterns=one.num_used_patterns,
            xis=one.xis,
            eigen_values=one.eigen_values,
            eigen_vectors=one.eigen_vectors,
            scaling_factors=one.scaling_factors
        )
        self.assertTrue(one == two)

    def test_eq_with_neq_xis(self):
        num_xs = 10
        num_xis = 200
        dimension = 3
        one = Results(
            densities=np.random.rand(num_xs),
            num_used_patterns=np.random.rand(num_xs),
            xis=np.random.rand(num_xis, dimension),
            eigen_values=np.random.rand(num_xis, dimension),
            eigen_vectors=np.random.rand(num_xis, dimension, dimension),
            scaling_factors=np.random.rand(num_xis)
        )
        two = Results(
            densities=one.densities,
            num_used_patterns=one.num_used_patterns,
            xis=np.random.rand(num_xis, dimension),
            eigen_values=one.eigen_values,
            eigen_vectors=one.eigen_vectors,
            scaling_factors=one.scaling_factors
        )
        self.assertFalse(one == two)

    def test_eq_with_neq_eigen_values(self):
        num_xs = 10
        num_xis = 200
        dimension = 3
        one = Results(
            densities=np.random.rand(num_xs),
            num_used_patterns=np.random.rand(num_xs),
            xis=np.random.rand(num_xis, dimension),
            eigen_values=np.random.rand(num_xis, dimension),
            eigen_vectors=np.random.rand(num_xis, dimension, dimension),
            scaling_factors=np.random.rand(num_xis)
        )
        two = Results(
            densities=one.densities,
            num_used_patterns=one.num_used_patterns,
            xis=one.xis,
            eigen_values=np.random.rand(num_xis, dimension),
            eigen_vectors=one.eigen_vectors,
            scaling_factors=one.scaling_factors
        )
        self.assertFalse(one == two)

    def test_eq_with_neq_eigen_vectors(self):
        num_xs = 10
        num_xis = 200
        dimension = 3
        one = Results(
            densities=np.random.rand(num_xs),
            num_used_patterns=np.random.rand(num_xs),
            xis=np.random.rand(num_xis, dimension),
            eigen_values=np.random.rand(num_xis, dimension),
            eigen_vectors=np.random.rand(num_xis, dimension, dimension),
            scaling_factors=np.random.rand(num_xis)
        )
        two = Results(
            densities=one.densities,
            num_used_patterns=one.num_used_patterns,
            xis=one.xis,
            eigen_values=one.eigen_values,
            eigen_vectors=np.random.rand(num_xis, dimension, dimension),
            scaling_factors=one.scaling_factors
        )
        self.assertFalse(one == two)

    def test_eq_with_neq_scaling_factors(self):
        num_xs = 10
        num_xis = 200
        dimension = 3
        one = Results(
            densities=np.random.rand(num_xs),
            num_used_patterns=np.random.rand(num_xs),
            xis=np.random.rand(num_xis, dimension),
            eigen_values=np.random.rand(num_xis, dimension),
            eigen_vectors=np.random.rand(num_xis, dimension, dimension),
            scaling_factors=np.random.rand(num_xis)
        )
        two = Results(
            densities=one.densities,
            num_used_patterns=one.num_used_patterns,
            xis=one.xis,
            eigen_values=one.eigen_values,
            eigen_vectors=one.eigen_vectors,
            scaling_factors=np.random.rand(num_xis)
        )
        self.assertFalse(one == two)

    def test_eq_with_none_xis(self):
        num_xs = 10
        num_xis = 200
        dimension = 3
        one = Results(
            densities=np.random.rand(num_xs),
            num_used_patterns=np.random.rand(num_xs),
            xis=np.random.rand(num_xis, dimension),
            eigen_values=np.random.rand(num_xis, dimension),
            eigen_vectors=np.random.rand(num_xis, dimension, dimension),
            scaling_factors=np.random.rand(num_xis)
        )
        two = Results(
            densities=one.densities,
            num_used_patterns=one.num_used_patterns,
            xis=None,
            eigen_values=one.eigen_values,
            eigen_vectors=one.eigen_vectors,
            scaling_factors=one.scaling_factors
        )
        self.assertFalse(one == two)

    def test_eq_with_none_eigen_values(self):
        num_xs = 10
        num_xis = 200
        dimension = 3
        one = Results(
            densities=np.random.rand(num_xs),
            num_used_patterns=np.random.rand(num_xs),
            xis=np.random.rand(num_xis, dimension),
            eigen_values=np.random.rand(num_xis, dimension),
            eigen_vectors=np.random.rand(num_xis, dimension, dimension),
            scaling_factors=np.random.rand(num_xis)
        )
        two = Results(
            densities=one.densities,
            num_used_patterns=one.num_used_patterns,
            xis=one.xis,
            eigen_values=None,
            eigen_vectors=one.eigen_vectors,
            scaling_factors=one.scaling_factors
        )
        self.assertFalse(one == two)

    def test_eq_with_none_eigen_vectors(self):
        num_xs = 10
        num_xis = 200
        dimension = 3
        one = Results(
            densities=np.random.rand(num_xs),
            num_used_patterns=np.random.rand(num_xs),
            xis=np.random.rand(num_xis, dimension),
            eigen_values=np.random.rand(num_xis, dimension),
            eigen_vectors=np.random.rand(num_xis, dimension, dimension),
            scaling_factors=np.random.rand(num_xis)
        )
        two = Results(
            densities=one.densities,
            num_used_patterns=one.num_used_patterns,
            xis=one.xis,
            eigen_values=one.eigen_values,
            eigen_vectors=None,
            scaling_factors=one.scaling_factors
        )
        self.assertFalse(one == two)

    def test_eq_with_none_scaling_factors(self):
        num_xs = 10
        num_xis = 200
        dimension = 3
        one = Results(
            densities=np.random.rand(num_xs),
            num_used_patterns=np.random.rand(num_xs),
            xis=np.random.rand(num_xis, dimension),
            eigen_values=np.random.rand(num_xis, dimension),
            eigen_vectors=np.random.rand(num_xis, dimension, dimension),
            scaling_factors=np.random.rand(num_xis)
        )
        two = Results(
            densities=one.densities,
            num_used_patterns=one.num_used_patterns,
            xis=one.xis,
            eigen_values=one.eigen_values,
            eigen_vectors=one.eigen_vectors,
            scaling_factors=None
        )
        self.assertFalse(one == two)

    def test_eq_with_different_shapes(self):
        num_xs = 10
        num_xis_1 = 200
        num_xis_2 = 100
        dimension = 3
        one = Results(
            densities=np.random.rand(num_xs),
            num_used_patterns=np.random.rand(num_xs),
            xis=np.random.rand(num_xis_1, dimension),
            eigen_values=np.random.rand(num_xis_1, dimension),
            eigen_vectors=np.random.rand(num_xis_1, dimension, dimension),
            scaling_factors=np.random.rand(num_xis_1)
        )
        two = Results(
            densities=one.densities,
            num_used_patterns=one.num_used_patterns,
            xis=np.random.rand(num_xis_2, dimension),
            eigen_values=np.random.rand(num_xis_2, dimension),
            eigen_vectors=np.random.rand(num_xis_2, dimension, dimension),
            scaling_factors=np.random.rand(num_xis_2)
        )
        self.assertFalse(one == two)

    def test_eq_with_different_shape_densities(self):
        one = Results(
            densities=np.random.rand(100),
            num_used_patterns=np.random.rand(100),
        )
        two = Results(
            densities=np.random.rand(200),
            num_used_patterns=np.random.rand(200),
        )
        self.assertFalse(one == two)


class Test_DensitiesValidator(TestCase):
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
        warnings.simplefilter("always")

    def test_validate_1(self):
        densities = np.array([
            0.1, 0.2, 0.3, 0.4, 0.51234567891011121314
        ])
        validator = _DensitiesValidator(data_set=self._data_set, densities=densities)
        actual = validator.validate()
        self.assertIsNone(actual)

    def test_validate_2(self):
        try:
            densities = np.array([
                1.0, 2.0, 3.0, 4.0
            ])
            validator = _DensitiesValidator(data_set=self._data_set, densities=densities)
            validator.validate()
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__one_result_per_pattern_1(self):
        densities = np.array([
            1.0, 2.0, 3.0, 4.0, 5.1234567891011121314
        ])
        validator = _DensitiesValidator(data_set=self._data_set, densities=densities)
        actual = validator._one_result_per_pattern()
        self.assertIsNone(actual)

    def test__one_result_per_pattern_2(self):
        try:
            densities = np.array([
                1.0, 2.0, 3.0, 4.0
            ])
            validator = _DensitiesValidator(data_set=self._data_set, densities=densities)
            validator._one_result_per_pattern()
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__one_result_per_pattern_3(self):
        try:
            densities = np.array([
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0
            ])
            validator = _DensitiesValidator(data_set=self._data_set, densities=densities)
            validator._one_result_per_pattern()
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__dont_check_results_per_pattern_if_bool_is_set(self):
            densities = np.array([
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
            ])
            validator = _DensitiesValidator(data_set=None, densities=densities)
            validator.validate()

    def test__results_is_1D_array_1(self):
        densities = np.array([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0
        ])
        validator = _DensitiesValidator(data_set=self._data_set, densities=densities)
        actual = validator._results_is_1D_array()
        self.assertIsNone(actual)

    def test__results_is_1D_array_2(self):
        try:
            densities = np.array([
                [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]
            ])
            validator = _DensitiesValidator(data_set=self._data_set, densities=densities)
            validator._results_is_1D_array()
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__results_are_densities_with_edge_cases(self):
        densities = np.array([
            0.0, 0.2, 0.33, 0.444, 0.55, 1.0
        ])
        validator = _DensitiesValidator(data_set=self._data_set, densities=densities)
        with warnings.catch_warnings(record=True) as w:
            validator._results_are_densities()
            if len(w):
                self.fail('The warning was triggered')

    def test__results_are_densities_with_only_valid_densities(self):
        densities = np.array([
            7.539699219e-05,
            1.240164051e-05,
            1.227518586e-05,
            7.288289757e-05,
            0.0001832763582
        ])
        validator = _DensitiesValidator(data_set=self._data_set, densities=densities)
        with warnings.catch_warnings(record=True) as w:
            validator._results_are_densities()
            if len(w):
                self.fail('The warning was triggered')

    def test__results_are_densities_with_invalid_densities(self):
        densities = np.array([
            7.539699219e-05,
            1.240164051e+05,
            1.227518586e-05,
            7.288289757e-05,
            5.0001832763582
        ])
        with warnings.catch_warnings(record=True) as w:
            validator = _DensitiesValidator(data_set=self._data_set, densities=densities)
            validator._results_are_densities()
            if not len(w):
                self.fail('The warning was not triggered')

    def test_validate_density_with_valid_density(self):
        density = 0.5
        _DensitiesValidator.validate_density(density)

    def test_validate_density_with_edge_case_lower_bound(self):
        density = 0.0
        _DensitiesValidator.validate_density(density)

    def test_validate_density_with_edge_case_upper_bound(self):
        density = 1.0
        _DensitiesValidator.validate_density(density)

    def test_validate_density_with_invalid_density(self):
        try:
            density = 1.5
            _DensitiesValidator.validate_density(density)
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')


class Test__XisValidator(TestCase):
    def test_xis_dimension(self):
        xis = np.random.rand(10, 3)
        eigen_values = None
        eigen_vectors = np.random.rand(10, 3, 3)

        actual = _XisValidator(
            xis=xis,
            eigen_values=eigen_values,
            eigen_vectors=eigen_vectors
        ).xis_dimension
        expected = 3
        self.assertEqual(actual, expected)

    def test_xis_count(self):
        xis = np.random.rand(10, 3)
        eigen_values = None
        eigen_vectors = np.random.rand(10, 3, 3)

        actual = _XisValidator(
            xis=xis,
            eigen_values=eigen_values,
            eigen_vectors=eigen_vectors
        ).xis_count
        expected = 10
        self.assertEqual(actual, expected)

    def test_no_eigen_values(self):
        xis = np.random.rand(10, 3)
        eigen_values = None
        eigen_vectors = np.random.rand(10, 3, 3)

        actual = _XisValidator(
            xis=xis,
            eigen_values=eigen_values,
            eigen_vectors=eigen_vectors
        ).validate()
        self.assertIsNone(actual)

    def test_no_eigen_vectors(self):
        xis = np.random.rand(10, 3)
        eigen_values = np.random.rand(10, 3)
        eigen_vectors = None

        actual = _XisValidator(
            xis=xis,
            eigen_values=eigen_values,
            eigen_vectors=eigen_vectors
        ).validate()
        self.assertIsNone(actual)

    def test_no_eigen_properties(self):
        xis = np.random.rand(10, 3)
        eigen_values = None
        eigen_vectors = None

        actual = _XisValidator(
            xis=xis,
            eigen_values=eigen_values,
            eigen_vectors=eigen_vectors
        ).validate()
        self.assertIsNone(actual)

    def test_no_scaling_factors(self):
        xis = np.random.rand(10, 3)
        eigen_values = np.random.rand(10, 3)
        eigen_vectors = np.random.rand(10, 3, 3)
        scaling_factors = None

        actual = _XisValidator(
            xis=xis,
            eigen_values=eigen_values,
            eigen_vectors=eigen_vectors,
            scaling_factors=scaling_factors
        ).validate()
        self.assertIsNone(actual)

    def test_wrong_number_of_eigen_values_too_many(self):
        try:
            xis = np.random.rand(10, 3)
            eigen_values = np.random.rand(20, 3)
            eigen_vectors = np.random.rand(10, 3, 3)

            actual = _XisValidator(
                xis=xis,
                eigen_vectors=eigen_vectors,
                eigen_values=eigen_values,
            ).validate()
            self.assertIsNone(actual)
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_wrong_number_of_eigen_values_too_few(self):
        try:
            xis = np.random.rand(10, 3)
            eigen_values = np.random.rand(4, 3)
            eigen_vectors = np.random.rand(10, 3, 3)

            actual = _XisValidator(
                xis=xis,
                eigen_vectors=eigen_vectors,
                eigen_values=eigen_values,
            ).validate()
            self.assertIsNone(actual)
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_eigen_values_wrong_ndim_too_few(self):
        try:
            xis = np.random.rand(10, 3)
            eigen_values = np.random.rand(10)
            eigen_vectors = np.random.rand(10, 3, 3)

            actual = _XisValidator(
                xis=xis,
                eigen_vectors=eigen_vectors,
                eigen_values=eigen_values,
            ).validate()
            self.assertIsNone(actual)
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_eigen_values_wrong_ndim_too_many(self):
        try:
            xis = np.random.rand(10, 3)
            eigen_values = np.random.rand(10, 3, 3)
            eigen_vectors = np.random.rand(10, 3, 3)

            actual = _XisValidator(
                xis=xis,
                eigen_vectors=eigen_vectors,
                eigen_values=eigen_values,
            ).validate()
            self.assertIsNone(actual)
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_wrong_number_of_eigen_values_per_pattern_too_few(self):
        try:
            xis = np.random.rand(10, 3)
            eigen_values = np.random.rand(10, 2)
            eigen_vectors = np.random.rand(10, 3, 3)

            actual = _XisValidator(
                xis=xis,
                eigen_vectors=eigen_vectors,
                eigen_values=eigen_values,
            ).validate()
            self.assertIsNone(actual)
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_wrong_number_of_eigen_values_per_pattern_too_many(self):
        try:
            xis = np.random.rand(10, 3)
            eigen_values = np.random.rand(10, 5)
            eigen_vectors = np.random.rand(10, 3, 3)

            actual = _XisValidator(
                xis=xis,
                eigen_vectors=eigen_vectors,
                eigen_values=eigen_values,
            ).validate()
            self.assertIsNone(actual)
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_wrong_number_of_eigen_vectors_too_many(self):
        try:
            xis = np.random.rand(10, 3)
            eigen_values = np.random.rand(10, 3)
            eigen_vectors = np.random.rand(20, 3, 3)

            actual = _XisValidator(
                xis=xis,
                eigen_vectors=eigen_vectors,
                eigen_values=eigen_values,
            ).validate()
            self.assertIsNone(actual)
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_wrong_number_of_eigen_vectors_too_few(self):
        try:
            xis = np.random.rand(10, 3)
            eigen_values = np.random.rand(10, 3)
            eigen_vectors = np.random.rand(5, 3, 3)

            actual = _XisValidator(
                xis=xis,
                eigen_vectors=eigen_vectors,
                eigen_values=eigen_values,
            ).validate()
            self.assertIsNone(actual)
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_wrong_number_of_eigen_vectors_per_xis_too_many(self):
        try:
            xis = np.random.rand(10, 3)
            eigen_values = np.random.rand(10, 3)
            eigen_vectors = np.random.rand(10, 5, 3)

            actual = _XisValidator(
                xis=xis,
                eigen_vectors=eigen_vectors,
                eigen_values=eigen_values,
            ).validate()
            self.assertIsNone(actual)
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_wrong_number_of_eigen_vectors_per_xis_too_few(self):
        try:
            xis = np.random.rand(10, 3)
            eigen_values = np.random.rand(10, 3)
            eigen_vectors = np.random.rand(10, 1, 3)

            actual = _XisValidator(
                xis=xis,
                eigen_vectors=eigen_vectors,
                eigen_values=eigen_values,
            ).validate()
            self.assertIsNone(actual)
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def tests_eigen_value_array_dimension_too_high(self):
        try:
            xis = np.random.rand(10, 3)
            eigen_values = np.random.rand(10, 3)
            eigen_vectors = np.random.rand(10, 3, 3, 4)

            actual = _XisValidator(
                xis=xis,
                eigen_vectors=eigen_vectors,
                eigen_values=eigen_values,
            ).validate()
            self.assertIsNone(actual)
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def tests_eigen_value_array_dimension_too_low(self):
        try:
            xis = np.random.rand(10, 3)
            eigen_values = np.random.rand(10, 3)
            eigen_vectors = np.random.rand(10, 3)

            actual = _XisValidator(
                xis=xis,
                eigen_vectors=eigen_vectors,
                eigen_values=eigen_values,
            ).validate()
            self.assertIsNone(actual)
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_wrong_number_of_eigen_values_for_dimension_too_many(self):
        try:
            xis = np.random.rand(10, 3)
            eigen_values = np.random.rand(10, 3)
            eigen_vectors = np.random.rand(10, 5, 3)

            actual = _XisValidator(
                xis=xis,
                eigen_vectors=eigen_vectors,
                eigen_values=eigen_values,
            ).validate()
            self.assertIsNone(actual)
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_wrong_number_of_eigen_values_for_dimension_too_few(self):
        try:
            xis = np.random.rand(10, 3)
            eigen_values = np.random.rand(10, 3)
            eigen_vectors = np.random.rand(10, 2, 3)

            actual = _XisValidator(
                xis=xis,
                eigen_vectors=eigen_vectors,
                eigen_values=eigen_values,
            ).validate()
            self.assertIsNone(actual)
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_wrong_eigen_vectors_dimension_too_many(self):
        try:
            xis = np.random.rand(10, 3)
            eigen_values = np.random.rand(10, 3)
            eigen_vectors = np.random.rand(10, 3, 5)

            actual = _XisValidator(
                xis=xis,
                eigen_vectors=eigen_vectors,
                eigen_values=eigen_values,
            ).validate()
            self.assertIsNone(actual)
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_wrong_eigen_vectors_dimension_too_few(self):
        try:
            xis = np.random.rand(10, 3)
            eigen_values = np.random.rand(10, 3)
            eigen_vectors = np.random.rand(10, 3, 2)

            actual = _XisValidator(
                xis=xis,
                eigen_vectors=eigen_vectors,
                eigen_values=eigen_values,
            ).validate()
            self.assertIsNone(actual)
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_wrong_number_of_scaling_factors_too_low(self):
        try:
            xis = np.random.rand(10, 3)
            scaling_factors = np.random.rand(5)
            eigen_values = np.random.rand(10, 3)
            eigen_vectors = np.random.rand(10, 3, 3)

            actual = _XisValidator(
                xis=xis,
                eigen_vectors=eigen_vectors,
                eigen_values=eigen_values,
                scaling_factors=scaling_factors,
            ).validate()
            self.assertIsNone(actual)
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_wrong_number_of_scaling_factors_too_high(self):
        try:
            xis = np.random.rand(10, 3)
            scaling_factors = np.random.rand(20)
            eigen_values = np.random.rand(10, 3)
            eigen_vectors = np.random.rand(10, 3, 3)

            actual = _XisValidator(
                xis=xis,
                eigen_vectors=eigen_vectors,
                eigen_values=eigen_values,
                scaling_factors=scaling_factors,
            ).validate()
            self.assertIsNone(actual)
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_wrong_scaling_factor_dimension_too_high(self):
        try:
            xis = np.random.rand(10, 3)
            scaling_factors = np.random.rand(20, 2)
            eigen_values = np.random.rand(10, 3)
            eigen_vectors = np.random.rand(10, 3, 3)

            actual = _XisValidator(
                xis=xis,
                eigen_vectors=eigen_vectors,
                eigen_values=eigen_values,
                scaling_factors=scaling_factors,
            ).validate()
            self.assertIsNone(actual)
        except InvalidResultsException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_valid_combination(self):
        xis = np.random.rand(10, 3)
        eigen_values = np.random.rand(10, 3)
        eigen_vectors = np.random.rand(10, 3, 3)
        scaling_factors = np.random.rand(10)

        actual = _XisValidator(
            xis=xis,
            eigen_vectors=eigen_vectors,
            eigen_values=eigen_values,
            scaling_factors=scaling_factors,
        ).validate()
        self.assertIsNone(actual)
