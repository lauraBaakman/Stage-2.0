from unittest import TestCase

from unipath import Path

import inputoutput.files as filenames


class TestIsPathFunctions(TestCase):
    def setUp(self):
        super(TestIsPathFunctions, self).setUp()
        self._grid_file_string = '/Users/laura/Repositories/baakman_1_600000_grid_128.txt'
        self._grid_file_path = Path(self._grid_file_string)

        self._grid_result_file_string_sens = '/Users/laura/Repositories/baakman_1_600000_mbe_breiman_grid_128.txt'
        self._grid_result_file_path_sens = Path(self._grid_result_file_string_sens)

        self._grid_result_file_string_no_sens = '/Users/laura/Repositories/baakman_1_600000_mbe_grid_128.txt'
        self._grid_result_file_path_no_sens = Path(self._grid_result_file_string_no_sens)

        self._dataset_file_string = '/Users/laura/Repositories/baakman_1_600000.txt'
        self._dataset_file_path = Path(self._dataset_file_string)

        self._dataset_result_file_string_sens = '/Users/laura/baakman_1_600000_mbe_breiman.txt'
        self._dataset_result_file_path_sens = Path(self._dataset_result_file_string_sens)

        self._dataset_result_file_string_no_sens = '/Users/laura/baakman_1_600000_mbe.txt'
        self._dataset_result_file_path_no_sens = Path(self._dataset_result_file_string_no_sens)

        self._xis_file_no_grid_string_sens = '/Users/laura/Repositories/baakman_1_600000_mbe_breiman_xis.txt'
        self._xis_file_no_grid_path_sens = Path(self._xis_file_no_grid_string_sens)

        self._xis_file_no_grid_string_no_sens = '/Users/laura/Repositories/baakman_1_600000_mbe_xis.txt'
        self._xis_file_no_grid_path_no_sens = Path(self._xis_file_no_grid_string_no_sens)

        self._xis_file_with_grid_string_sens = '/Users/laura/Repositories/baakman_1_600000_mbe_breiman_grid_128_xis.txt'
        self._xis_file_with_grid_path_sens = Path(self._xis_file_with_grid_string_sens)

        self._xis_file_with_grid_string_no_sens = '/Users/laura/Repositories/baakman_1_600000_mbe_grid_128_xis.txt'
        self._xis_file_with_grid_path_no_sens = Path(self._xis_file_with_grid_string_no_sens)

    # Test for is grid_file
    def test_is_grid_file_grid_file(self):
        self.assertTrue(filenames.is_grid_file(self._grid_file_string))
        self.assertTrue(filenames.is_grid_file(self._grid_file_path))

    def test_is_grid_file_grid_result_file_sens(self):
        self.assertFalse(filenames.is_grid_file(self._grid_result_file_string_sens))
        self.assertFalse(filenames.is_grid_file(self._grid_result_file_path_sens))

    def test_is_grid_file_grid_result_file_no_sens(self):
        self.assertFalse(filenames.is_grid_file(self._grid_result_file_string_no_sens))
        self.assertFalse(filenames.is_grid_file(self._grid_result_file_path_no_sens))

    def test_is_grid_file_dataset_file(self):
        self.assertFalse(filenames.is_grid_file(self._dataset_file_string))
        self.assertFalse(filenames.is_grid_file(self._dataset_file_string))

    def test_is_grid_file_dataset_result_file_sens(self):
        self.assertFalse(filenames.is_grid_file(self._dataset_result_file_string_sens))
        self.assertFalse(filenames.is_grid_file(self._dataset_result_file_path_sens))

    def test_is_grid_file_dataset_result_file_no_sens(self):
        self.assertFalse(filenames.is_grid_file(self._dataset_result_file_string_no_sens))
        self.assertFalse(filenames.is_grid_file(self._dataset_result_file_path_no_sens))

    def test_is_grid_file_xis_file_no_grid_sens(self):
        self.assertFalse(filenames.is_grid_file(self._xis_file_no_grid_string_sens))
        self.assertFalse(filenames.is_grid_file(self._xis_file_no_grid_path_sens))

    def test_is_grid_file_xis_no_grid_no_sens(self):
        self.assertFalse(filenames.is_grid_file(self._xis_file_no_grid_string_no_sens))
        self.assertFalse(filenames.is_grid_file(self._xis_file_no_grid_path_no_sens))

    def test_is_grid_file_xis_file_with_grid_sens(self):
        self.assertFalse(filenames.is_grid_file(self._xis_file_with_grid_string_sens))
        self.assertFalse(filenames.is_grid_file(self._xis_file_with_grid_path_sens))

    def test_is_grid_file_xis_file_with_grid_no_sens(self):
        self.assertFalse(filenames.is_grid_file(self._xis_file_with_grid_string_no_sens))
        self.assertFalse(filenames.is_grid_file(self._xis_file_with_grid_path_no_sens))

    # Tests for is_grid_result_file
    def test_is_grid_result_file_grid_file(self):
        self.assertFalse(filenames.is_grid_result_file(self._grid_file_string))
        self.assertFalse(filenames.is_grid_result_file(self._grid_file_path))

    def test_is_grid_result_file_grid_result_file_sens(self):
        self.assertTrue(filenames.is_grid_result_file(self._grid_result_file_string_sens))
        self.assertTrue(filenames.is_grid_result_file(self._grid_result_file_path_sens))

    def test_is_grid_result_file_grid_result_file_no_sens(self):
        self.assertTrue(filenames.is_grid_result_file(self._grid_result_file_string_no_sens))
        self.assertTrue(filenames.is_grid_result_file(self._grid_result_file_path_no_sens))

    def test_is_grid_result_file_dataset_file(self):
        self.assertFalse(filenames.is_grid_result_file(self._dataset_file_string))
        self.assertFalse(filenames.is_grid_result_file(self._dataset_file_string))

    def test_is_grid_result_file_dataset_result_file_sens(self):
        self.assertFalse(filenames.is_grid_result_file(self._dataset_result_file_string_sens))
        self.assertFalse(filenames.is_grid_result_file(self._dataset_result_file_path_sens))

    def test_is_grid_result_file_dataset_result_file_no_sens(self):
        self.assertFalse(filenames.is_grid_result_file(self._dataset_result_file_string_no_sens))
        self.assertFalse(filenames.is_grid_result_file(self._dataset_result_file_path_no_sens))

    def test_is_grid_result_file_xis_file_no_grid_no_sens(self):
        self.assertFalse(filenames.is_grid_result_file(self._xis_file_no_grid_string_no_sens))
        self.assertFalse(filenames.is_grid_result_file(self._xis_file_no_grid_path_no_sens))

    def test_is_grid_result_file_xis_file_no_grid_with_sens(self):
        self.assertFalse(filenames.is_grid_result_file(self._xis_file_with_grid_string_sens))
        self.assertFalse(filenames.is_grid_result_file(self._xis_file_with_grid_path_sens))

    def test_is_grid_result_file_xis_file_with_grid_no_sens(self):
        self.assertFalse(filenames.is_grid_result_file(self._xis_file_with_grid_string_no_sens))
        self.assertFalse(filenames.is_grid_result_file(self._xis_file_with_grid_path_no_sens))

    def test_is_grid_result_file_xis_file_with_grid_with_sens(self):
        self.assertFalse(filenames.is_grid_result_file(self._xis_file_with_grid_string_sens))
        self.assertFalse(filenames.is_grid_result_file(self._xis_file_with_grid_path_sens))

    # Tests for is_dataset_file
    def test_is_dataset_file_xis_file_no_grid_no_sens(self):
        self.assertFalse(filenames.is_dataset_file(self._xis_file_no_grid_string_no_sens))
        self.assertFalse(filenames.is_dataset_file(self._xis_file_no_grid_path_sens))

    def test_is_dataset_file_grid_result_file_sens(self):
        self.assertFalse(filenames.is_dataset_file(self._grid_result_file_string_sens))
        self.assertFalse(filenames.is_dataset_file(self._grid_result_file_path_sens))

    def test_is_dataset_file_grid_result_file_no_sens(self):
        self.assertFalse(filenames.is_dataset_file(self._grid_result_file_string_no_sens))
        self.assertFalse(filenames.is_dataset_file(self._grid_result_file_path_no_sens))

    def test_is_dataset_file_dataset_file(self):
        self.assertTrue(filenames.is_dataset_file(self._dataset_file_string))
        self.assertTrue(filenames.is_dataset_file(self._dataset_file_string))

    def test_is_dataset_file_dataset_result_file_sens(self):
        self.assertFalse(filenames.is_dataset_file(self._dataset_result_file_string_sens))
        self.assertFalse(filenames.is_dataset_file(self._dataset_result_file_path_sens))

    def test_is_dataset_file_dataset_result_file_no_sens(self):
        self.assertFalse(filenames.is_dataset_file(self._dataset_result_file_string_no_sens))
        self.assertFalse(filenames.is_dataset_file(self._dataset_result_file_path_no_sens))

    def test_is_dataset_file_dataset_xis_file_no_grid_sens(self):
        self.assertFalse(filenames.is_dataset_file(self._xis_file_no_grid_string_sens))
        self.assertFalse(filenames.is_dataset_file(self._xis_file_no_grid_path_sens))

    def test_is_dataset_file_dataset_xis_file_no_grid_no_sens(self):
        self.assertFalse(filenames.is_dataset_file(self._xis_file_no_grid_string_no_sens))
        self.assertFalse(filenames.is_dataset_file(self._xis_file_no_grid_path_no_sens))

    def test_is_dataset_file_dataset_xis_file_with_grid_sens(self):
        self.assertFalse(filenames.is_dataset_file(self._xis_file_with_grid_string_sens))
        self.assertFalse(filenames.is_dataset_file(self._xis_file_with_grid_path_sens))

    def test_is_dataset_file_dataset_xis_file_with_grid_no_sens(self):
        self.assertFalse(filenames.is_dataset_file(self._xis_file_with_grid_string_no_sens))
        self.assertFalse(filenames.is_dataset_file(self._xis_file_with_grid_path_no_sens))

    # Tests for is_dataset_result_file
    def test_is_dataset_result_file_grid_file(self):
        self.assertFalse(filenames.is_dataset_result_file(self._grid_file_string))
        self.assertFalse(filenames.is_dataset_result_file(self._grid_file_path))

    def test_is_dataset_result_file_grid_result_file_sens(self):
        self.assertFalse(filenames.is_dataset_result_file(self._grid_result_file_string_sens))
        self.assertFalse(filenames.is_dataset_result_file(self._grid_result_file_path_sens))

    def test_is_dataset_result_file_grid_result_file_no_sens(self):
        self.assertFalse(filenames.is_dataset_result_file(self._grid_result_file_string_no_sens))
        self.assertFalse(filenames.is_dataset_result_file(self._grid_result_file_path_no_sens))

    def test_is_dataset_result_file_dataset_file(self):
        self.assertFalse(filenames.is_dataset_result_file(self._dataset_file_string))
        self.assertFalse(filenames.is_dataset_result_file(self._dataset_file_string))

    def test_is_dataset_result_file_dataset_result_file_sens(self):
        self.assertTrue(filenames.is_dataset_result_file(self._dataset_result_file_string_sens))
        self.assertTrue(filenames.is_dataset_result_file(self._dataset_result_file_path_sens))

    def test_is_dataset_result_file_dataset_result_file_no_sens(self):
        self.assertTrue(filenames.is_dataset_result_file(self._dataset_result_file_string_no_sens))
        self.assertTrue(filenames.is_dataset_result_file(self._dataset_result_file_path_no_sens))

    def test_is_dataset_result_file_xis_file_no_grid_string_sens(self):
        self.assertFalse(filenames.is_dataset_result_file(self._xis_file_no_grid_string_sens))
        self.assertFalse(filenames.is_dataset_result_file(self._xis_file_no_grid_path_sens))

    def test_is_dataset_result_file__xis_file_no_grid_no_sens(self):
        self.assertFalse(filenames.is_dataset_result_file(self._xis_file_no_grid_string_no_sens))
        self.assertFalse(filenames.is_dataset_result_file(self._xis_file_no_grid_path_no_sens))

    def test_is_dataset_result_file_xis_file_with_grid_sens(self):
        self.assertFalse(filenames.is_dataset_result_file(self._xis_file_with_grid_string_sens))
        self.assertFalse(filenames.is_dataset_result_file(self._xis_file_with_grid_path_sens))

    def test_is_dataset_result_file_xis_file_with_grid_no_sens(self):
        self.assertFalse(filenames.is_dataset_result_file(self._xis_file_with_grid_string_no_sens))
        self.assertFalse(filenames.is_dataset_result_file(self._xis_file_with_grid_path_no_sens))

    # Tests for is_xs_file
    def test_is_xs_file_grid_file(self):
        self.assertTrue(filenames.is_xs_file(self._grid_file_string))
        self.assertTrue(filenames.is_xs_file(self._grid_file_path))

    def test_is_xs_file_grid_result_file_sens(self):
        self.assertFalse(filenames.is_xs_file(self._grid_result_file_string_sens))
        self.assertFalse(filenames.is_xs_file(self._grid_result_file_path_sens))

    def test_is_xs_file_grid_result_file_no_sens(self):
        self.assertFalse(filenames.is_xs_file(self._grid_result_file_string_no_sens))
        self.assertFalse(filenames.is_xs_file(self._grid_result_file_path_no_sens))

    def test_is_xs_file_dataset_file(self):
        self.assertTrue(filenames.is_xs_file(self._dataset_file_string))
        self.assertTrue(filenames.is_xs_file(self._dataset_file_string))

    def test_is_xs_file_dataset_result_file_sens(self):
        self.assertFalse(filenames.is_xs_file(self._dataset_result_file_string_sens))
        self.assertFalse(filenames.is_xs_file(self._dataset_result_file_path_sens))

    def test_is_xs_file_dataset_result_file_no_sens(self):
        self.assertFalse(filenames.is_xs_file(self._dataset_result_file_string_no_sens))
        self.assertFalse(filenames.is_xs_file(self._dataset_result_file_path_no_sens))

    def test_is_xs_file_xis_file_no_grid_sens(self):
        self.assertFalse(filenames.is_xs_file(self._xis_file_no_grid_string_sens))
        self.assertFalse(filenames.is_xs_file(self._xis_file_no_grid_string_sens))

    def test_is_xs_file_xis_file_no_grid_no_sens(self):
        self.assertFalse(filenames.is_xs_file(self._xis_file_no_grid_string_no_sens))
        self.assertFalse(filenames.is_xs_file(self._xis_file_no_grid_path_no_sens))

    def test_is_xs_file_xis_file_with_grid_sens(self):
        self.assertFalse(filenames.is_xs_file(self._xis_file_with_grid_string_sens))
        self.assertFalse(filenames.is_xs_file(self._xis_file_with_grid_path_sens))

    def test_is_xs_file_xis_file_with_grid_no_sens(self):
        self.assertFalse(filenames.is_xs_file(self._xis_file_with_grid_string_no_sens))
        self.assertFalse(filenames.is_xs_file(self._xis_file_with_grid_path_no_sens))

    # Tests for is_results_file
    def test_is_results_file_grid_file(self):
        self.assertFalse(filenames.is_results_file(self._grid_file_string))
        self.assertFalse(filenames.is_results_file(self._grid_file_path))

    def test_is_results_file_grid_result_file_sens(self):
        self.assertTrue(filenames.is_results_file(self._grid_result_file_string_sens))
        self.assertTrue(filenames.is_results_file(self._grid_result_file_path_sens))

    def test_is_results_file_grid_result_file_no_sens(self):
        self.assertTrue(filenames.is_results_file(self._grid_result_file_string_no_sens))
        self.assertTrue(filenames.is_results_file(self._grid_result_file_path_no_sens))

    def test_is_results_file_dataset_file(self):
        self.assertFalse(filenames.is_results_file(self._dataset_file_string))
        self.assertFalse(filenames.is_results_file(self._dataset_file_string))

    def test_is_results_file_dataset_result_file_sens(self):
        self.assertTrue(filenames.is_results_file(self._dataset_result_file_string_sens))
        self.assertTrue(filenames.is_results_file(self._dataset_result_file_path_sens))

    def test_is_results_file_dataset_result_file_no_sens(self):
        self.assertTrue(filenames.is_results_file(self._dataset_result_file_string_no_sens))
        self.assertTrue(filenames.is_results_file(self._dataset_result_file_path_no_sens))

    def test_is_results_file_xis_file_no_grid_sens(self):
        self.assertFalse(filenames.is_results_file(self._xis_file_no_grid_string_sens))
        self.assertFalse(filenames.is_results_file(self._xis_file_no_grid_path_sens))

    def test_is_results_file_xis_file_no_grid_no_sens(self):
        self.assertFalse(filenames.is_results_file(self._xis_file_no_grid_string_no_sens))
        self.assertFalse(filenames.is_results_file(self._xis_file_no_grid_path_no_sens))

    def test_is_results_file_xis_file_with_grid_sens(self):
        self.assertFalse(filenames.is_results_file(self._xis_file_with_grid_string_sens))
        self.assertFalse(filenames.is_results_file(self._xis_file_with_grid_path_sens))

    def test_is_results_file_xis_file_with_grid_no_sens(self):
        self.assertFalse(filenames.is_results_file(self._xis_file_with_grid_string_no_sens))
        self.assertFalse(filenames.is_results_file(self._xis_file_with_grid_path_no_sens))

    # Tests for is_xis_files
    def test_is_xis_files_grid_file_string(self):
        self.assertFalse(filenames.is_xis_file(self._grid_file_string))
        self.assertFalse(filenames.is_xis_file(self._grid_file_path))

    def test_is_xis_files_grid_result_file_string_sens(self):
        self.assertFalse(filenames.is_xis_file(self._grid_result_file_string_sens))
        self.assertFalse(filenames.is_xis_file(self._grid_result_file_path_sens))

    def test_is_xis_files_grid_result_file_string_no_sens(self):
        self.assertFalse(filenames.is_xis_file(self._grid_result_file_string_no_sens))
        self.assertFalse(filenames.is_xis_file(self._grid_result_file_path_no_sens))

    def test_is_xis_files_dataset_file_string(self):
        self.assertFalse(filenames.is_xis_file(self._dataset_file_string))
        self.assertFalse(filenames.is_xis_file(self._dataset_file_path))

    def test_is_xis_files_dataset_result_file_string_sens(self):
        self.assertFalse(filenames.is_xis_file(self._dataset_result_file_string_sens))
        self.assertFalse(filenames.is_xis_file(self._dataset_result_file_path_sens))

    def test_is_xis_files_dataset_result_file_string_no_sens(self):
        self.assertFalse(filenames.is_xis_file(self._dataset_result_file_string_no_sens))
        self.assertFalse(filenames.is_xis_file(self._dataset_result_file_path_no_sens))

    def test_is_xis_files_xis_file_no_grid_string_sens(self):
        self.assertTrue(filenames.is_xis_file(self._xis_file_no_grid_string_sens))
        self.assertTrue(filenames.is_xis_file(self._xis_file_no_grid_path_sens))

    def test_is_xis_files_xis_file_no_grid_string_no_sens(self):
        self.assertTrue(filenames.is_xis_file(self._xis_file_no_grid_string_no_sens))
        self.assertTrue(filenames.is_xis_file(self._xis_file_no_grid_path_no_sens))

    def test_is_xis_files_xis_file_with_grid_string_sens(self):
        self.assertTrue(filenames.is_xis_file(self._xis_file_with_grid_string_sens))
        self.assertTrue(filenames.is_xis_file(self._xis_file_with_grid_path_sens))

    def test_is_xis_files_xis_file_with_grid_string_no_sens(self):
        self.assertTrue(filenames.is_xis_file(self._xis_file_with_grid_string_no_sens))
        self.assertTrue(filenames.is_xis_file(self._xis_file_with_grid_path_no_sens))


class TestParsePath(TestCase):

    def test_split_grid_file(self):
        path = '/Users/laura/Repositories/baakman_1_600000_grid_128.txt'
        actual = filenames.parse_path(path)
        expected = {
            'semantic_name': 'baakman_1',
            'size': 600000,
            'grid_size': 128
        }
        self.assertEqual(actual, expected)

    def test_split_grid_result_file_sens(self):
        path = '/Users/laura/Repositories/baakman_1_600000_mbe_breiman_grid_128.txt'
        actual = filenames.parse_path(path)
        expected = None
        expected = {
            'semantic_name': 'baakman_1',
            'size': 600000,
            'estimator': 'mbe',
            'sensitivity': 'breiman',
            'grid_size': 128
        }
        self.assertEqual(actual, expected)

    def test_split_grid_result_file_no_sens(self):
        path = '/Users/laura/Repositories/baakman_1_600000_mbe_grid_128.txt'
        actual = filenames.parse_path(path)
        expected = {
            'semantic_name': 'baakman_1',
            'size': 600000,
            'estimator': 'mbe',
            'sensitivity': None,
            'grid_size': 128
        }
        self.assertEqual(actual, expected)

    def test_split_dataset_file(self):
        path = '/Users/laura/Repositories/baakman_1_600000.txt'
        actual = filenames.parse_path(path)
        expected = {
            'semantic_name': 'baakman_1',
            'size': 600000,
        }
        self.assertEqual(actual, expected)

    def test_split_dataset_result_file_sens(self):
        path = '/Users/laura/baakman_1_600000_mbe_breiman.txt'
        actual = filenames.parse_path(path)
        expected = {
            'semantic_name': 'baakman_1',
            'size': 600000,
            'estimator': 'mbe',
            'sensitivity': 'breiman'
        }
        self.assertEqual(actual, expected)

    def test_split_dataset_result_file_no_sens(self):
        path = '/Users/laura/baakman_1_600000_mbe.txt'
        actual = filenames.parse_path(path)
        expected = {
            'semantic_name': 'baakman_1',
            'size': 600000,
            'estimator': 'mbe',
            'sensitivity': None
        }
        self.assertEqual(actual, expected)

    def test_split_xis_file_no_grid_no_sens(self):
        path = '/Users/laura/Repositories/baakman_1_600000_mbe_xis.txt'
        actual = filenames.parse_path(path)
        expected = {
            'semantic_name': 'baakman_1',
            'size': 600000,
            'estimator': 'mbe',
            'sensitivity': None,
            'contains xis data': True,
        }
        self.assertEqual(actual, expected)

    def test_split_xis_file_no_grid_sens(self):
        path = '/Users/laura/Repositories/baakman_1_600000_mbe_breiman_xis.txt'
        actual = filenames.parse_path(path)
        expected = {
            'semantic_name': 'baakman_1',
            'size': 600000,
            'estimator': 'mbe',
            'sensitivity': 'breiman',
            'contains xis data': True,
        }
        self.assertEqual(actual, expected)

    def test_split_xis_file_grid_no_sens(self):
        path = '/Users/laura/Repositories/baakman_1_600000_mbe_grid_128_xis.txt'
        actual = filenames.parse_path(path)
        expected = {
            'semantic_name': 'baakman_1',
            'size': 600000,
            'estimator': 'mbe',
            'sensitivity': None,
            'contains xis data': True,
            'grid_size': 128
        }
        self.assertEqual(actual, expected)

    def test_split_xis_file_grid_sens(self):
        path = '/Users/laura/Repositories/baakman_1_600000_mbe_breiman_grid_128_xis.txt'
        actual = filenames.parse_path(path)
        expected = {
            'semantic_name': 'baakman_1',
            'size': 600000,
            'estimator': 'mbe',
            'sensitivity': 'breiman',
            'contains xis data': True,
            'grid_size': 128
        }
        self.assertEqual(actual, expected)


class TestIsAssociatedFile(TestCase):

    def setUp(self):
        super(TestIsAssociatedFile, self).setUp()
        self._grid_meta = {
            'semantic_name': 'baakman_1',
            'size': 600000,
            'grid_size': 128
        }
        self._grid_results_meta = {
            'semantic_name': 'baakman_1',
            'size': 600000,
            'estimator': 'mbe',
            'sensitivity': 'breiman',
            'grid_size': 128
        }
        self._grid_results_meta_2 = {
            'semantic_name': 'baakman_2',
            'size': 600000,
            'estimator': 'mbe',
            'sensitivity': 'breiman',
            'grid_size': 128
        }
        self._dataset_meta = {
            'semantic_name': 'baakman_1',
            'size': 600000,
        }
        self._dataset_results_meta = {
            'semantic_name': 'baakman_1',
            'size': 600000,
            'estimator': 'mbe',
            'sensitivity': None
        }
        self._dataset_results_meta_2 = {
            'semantic_name': 'baakman_2',
            'size': 600000,
            'estimator': 'mbe',
            'sensitivity': 'breiman'
        }

        self._xis_meta_1 = {
            'semantic_name': 'baakman_1',
            'size': 600000,
            'estimator': 'mbe',
            'sensitivity': None,
            'contains xis data': True,
        }

        self._xis_meta_2 = {
            'semantic_name': 'baakman_1',
            'size': 600000,
            'estimator': 'mbe',
            'sensitivity': 'breiman',
            'contains xis data': True,
        }

        self._xis_meta_3 = {
            'semantic_name': 'baakman_1',
            'size': 600000,
            'estimator': 'mbe',
            'sensitivity': None,
            'contains xis data': True,
            'grid_size': 128
        }

        self._xis_meta_4 = {
            'semantic_name': 'baakman_1',
            'size': 600000,
            'estimator': 'mbe',
            'sensitivity': 'breiman',
            'contains xis data': True,
            'grid_size': 128
        }

    def test_grid_grid_results_1(self):
        self.assertTrue(
            filenames.is_associated_result_file(
                self._grid_meta,
                self._grid_results_meta
            )
        )

    def test_grid_grid_results_2(self):
        self.assertFalse(
            filenames.is_associated_result_file(
                self._grid_meta,
                self._grid_results_meta_2
            )
        )

    def test_grid_dataset(self):
        self.assertFalse(
            filenames.is_associated_result_file(
                self._grid_meta,
                self._dataset_meta
            )
        )

    def test_grid_dataset_results_1(self):
        self.assertFalse(
            filenames.is_associated_result_file(
                self._grid_meta,
                self._dataset_results_meta
            )
        )

    def test_grid_dataset_results_2(self):
        self.assertFalse(
            filenames.is_associated_result_file(
                self._grid_meta,
                self._dataset_results_meta_2
            )
        )

    def test_dataset_grid_results_1(self):
        self.assertFalse(
            filenames.is_associated_result_file(
                self._dataset_meta,
                self._grid_results_meta
            )
        )

    def test_dataset_grid_results_2(self):
        self.assertFalse(
            filenames.is_associated_result_file(
                self._dataset_meta,
                self._grid_results_meta_2
            )
        )

    def test_dataset_dataset_results_1(self):
        self.assertTrue(
            filenames.is_associated_result_file(
                self._dataset_meta,
                self._dataset_results_meta
            )
        )

    def test_dataset_dataset_results_2(self):
        self.assertFalse(
            filenames.is_associated_result_file(
                self._dataset_meta,
                self._dataset_results_meta_2
            )
        )

    def test_dataset_dataset_results_skip_keys_1(self):
        self._dataset_meta.update({'file name': 'fjdkjflksjdflajslfa'})
        self._dataset_results_meta.update({'file name': 'jkdjflasjdflas'})
        self.assertFalse(
            filenames.is_associated_result_file(
                self._dataset_meta,
                self._dataset_results_meta_2,
                skip_keys=['file name']
            )
        )

    def test_dataset_dataset_results_skip_keys_2(self):
        self.assertTrue(
            filenames.is_associated_result_file(
                self._dataset_meta,
                self._dataset_results_meta_2,
                skip_keys=['semantic_name']
            )
        )

    def test_dataset_meta_xis_meta_1(self):
        self.assertTrue(
            filenames.is_associated_xis_file(
                self._dataset_meta,
                self._xis_meta_1
            )
        )

    def test_dataset_meta_xis_meta_2(self):
        self.assertTrue(
            filenames.is_associated_xis_file(
                self._dataset_meta,
                self._xis_meta_2
            )
        )

    def test_dataset_meta_xis_meta_3(self):
        self.assertFalse(
            filenames.is_associated_xis_file(
                self._dataset_meta,
                self._xis_meta_3
            )
        )

    def test_dataset_meta_xis_meta_4(self):
        self.assertFalse(
            filenames.is_associated_xis_file(
                self._dataset_meta,
                self._xis_meta_4
            )
        )

    def test_grid_meta_xis_meta_1(self):
        self.assertFalse(
            filenames.is_associated_xis_file(
                self._grid_meta,
                self._xis_meta_1
            )
        )

    def test_grid_meta_xis_meta_2(self):
        self.assertFalse(
            filenames.is_associated_xis_file(
                self._grid_meta,
                self._xis_meta_2
            )
        )

    def test_grid_meta_xis_meta_3(self):
        self.assertTrue(
            filenames.is_associated_xis_file(
                self._grid_meta,
                self._xis_meta_3
            )
        )

    def test_grid_meta_xis_meta_4(self):
        self.assertTrue(
            filenames.is_associated_xis_file(
                self._grid_meta,
                self._xis_meta_4
            )
        )

    def test_is_associated_xis_file_different_keys_but_same_number_of_keys(self):
        meta_1 = {
            'semantic_name': u'baakman_1',
            'grid_size': 64,
            'estimator': u'mbe',
            'sensitivity': u'breiman',
            'size': 60000
        }
        meta_2 = {
            'contains xis data': True,
            'semantic_name': u'baakman_1',
            'sensitivity': u'breiman',
            'estimator': u'mbe',
            'size': 60000
        }
        self.assertFalse(
            filenames.is_associated_xis_file(
                meta_1,
                meta_2
            )
        )
