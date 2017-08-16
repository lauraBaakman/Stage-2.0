from unittest import TestCase

from unipath import Path
import numpy as np

from inputoutput.utils import build_x_result_data_path, build_xs_path, sub_sample_grid, build_xi_result_data_path


class TestBuildResultPath(TestCase):

    def setUp(self):
        super(TestBuildResultPath, self).setUp()
        self._directory = Path('/Users/laura/Repositories/stage/data/simulated/small')
        self._data_set_file = Path('/Users/laura/Repositories/stage/data/simulated/small/ferdosi_1_90.txt')
        self._estimator = 'sambe'

    def test_no_sensitivity(self):
        actual = build_x_result_data_path(self._directory, self._data_set_file, self._estimator)
        expected = Path('/Users/laura/Repositories/stage/data/simulated/small/ferdosi_1_90_sambe.txt')
        self.assertEqual(actual, expected)

    def test_with_sensitivity(self):
        sensitivity = 'silverman'
        actual = build_x_result_data_path(self._directory, self._data_set_file, self._estimator, sensitivity)
        expected = Path('/Users/laura/Repositories/stage/data/simulated/small/ferdosi_1_90_sambe_silverman.txt')
        self.assertEqual(actual, expected)

    def test_with_grid(self):
        sensitivity = 'silverman'
        grid = 'grid256'
        actual = build_x_result_data_path(self._directory, self._data_set_file, self._estimator, sensitivity, grid)
        expected = Path(
            '/Users/laura/Repositories/stage/data/simulated/small/ferdosi_1_90_sambe_silverman_grid256.txt'
        )
        self.assertEqual(actual, expected)

    def test_with_empty_list(self):
        actual = build_x_result_data_path(self._directory, self._data_set_file, self._estimator, '')
        expected = Path('/Users/laura/Repositories/stage/data/simulated/small/ferdosi_1_90_sambe.txt')
        self.assertEqual(actual, expected)


class TestBuildXSPath(TestCase):
    def setUp(self):
        super(TestBuildXSPath, self).setUp()
        self._directory = Path('/Users/laura/Repositories/stage/data/simulated/small')
        self._data_set_file = Path('/Users/laura/Repositories/stage/data/simulated/small/ferdosi_1_90.txt')

    def test_no_args(self):
        actual = build_xs_path(self._directory, self._data_set_file)
        expected = Path(
            '/Users/laura/Repositories/stage/data/simulated/small/ferdosi_1_90.txt'
        )
        self.assertEqual(actual, expected)

    def test_with_args(self):
        grid_string = 'grid_128'
        actual = build_xs_path(self._directory, self._data_set_file, grid_string)
        expected = Path(
            '/Users/laura/Repositories/stage/data/simulated/small/ferdosi_1_90_grid_128.txt'
        )
        self.assertEqual(actual, expected)


class TestBuildXIResultPath(TestCase):
    def setUp(self):
        super(TestBuildXIResultPath, self).setUp()
        self._directory = Path('/Users/laura/Repositories/stage/data/simulated/small')
        self._data_set_file = Path('/Users/laura/Repositories/stage/data/simulated/small/ferdosi_1_90.txt')
        self._estimator = 'sambe'

    def test_no_sensitivity(self):
        actual = build_xi_result_data_path(self._directory, self._data_set_file, self._estimator)
        expected = Path('/Users/laura/Repositories/stage/data/simulated/small/ferdosi_1_90_sambe_xis.txt')
        self.assertEqual(actual, expected)

    def test_with_sensitivity(self):
        sensitivity = 'silverman'
        actual = build_xi_result_data_path(self._directory, self._data_set_file, self._estimator, sensitivity)
        expected = Path('/Users/laura/Repositories/stage/data/simulated/small/ferdosi_1_90_sambe_silverman_xis.txt')
        self.assertEqual(actual, expected)

    def test_with_grid(self):
        sensitivity = 'silverman'
        grid = 'grid256'
        actual = build_xi_result_data_path(self._directory, self._data_set_file, self._estimator, sensitivity, grid)
        expected = Path(
            '/Users/laura/Repositories/stage/data/simulated/small/ferdosi_1_90_sambe_silverman_grid256_xis.txt'
        )
        self.assertEqual(actual, expected)

    def test_with_empty_list(self):
        actual = build_xi_result_data_path(self._directory, self._data_set_file, self._estimator, '')
        expected = Path('/Users/laura/Repositories/stage/data/simulated/small/ferdosi_1_90_sambe_xis.txt')
        self.assertEqual(actual, expected)


class TestSubSampleGrid(TestCase):
    def test_sub_sample_square_grid(self):
        grid = np.array([
            [0.,  0.], [0.,  1.], [0.,  2.], [0.,  3.], [0.,  4.],
            [1.,  0.], [1.,  1.], [1.,  2.], [1.,  3.], [1.,  4.],
            [2.,  0.], [2.,  1.], [2.,  2.], [2.,  3.], [2.,  4.],
            [3.,  0.], [3.,  1.], [3.,  2.], [3.,  3.], [3.,  4.],
            [4.,  0.], [4.,  1.], [4.,  2.], [4.,  3.], [4.,  4.]
        ])
        expected = np.array([
           [0.,  0.], [0.,  2.], [0.,  4.],
           [2.,  0.], [2.,  2.], [2.,  4.],
           [4.,  0.], [4.,  2.], [4.,  4.]
        ])
        dimension = 2
        actual = sub_sample_grid(grid, space=1, dimension=dimension)
        np.testing.assert_array_equal(actual, expected)

    def test_sub_sample_rectangular_grid(self):
        grid = np.array([
            [0.,  0.], [1.,  0.], [2.,  0.], [3.,  0.], [4.,  0.],
            [0.,  1.], [1.,  1.], [2.,  1.], [3.,  1.], [4.,  1.],
            [0.,  2.], [1.,  2.], [2.,  2.], [3.,  2.], [4.,  2.],
            [0.,  3.], [1.,  3.], [2.,  3.], [3.,  3.], [4.,  3.],
            [0.,  4.], [1.,  4.], [2.,  4.], [3.,  4.], [4.,  4.],
            [0.,  5.], [1.,  5.], [2.,  5.], [3.,  5.], [4.,  5.],
            [0.,  6.], [1.,  6.], [2.,  6.], [3.,  6.], [4.,  6.]
        ])
        expected = np.array([
            [0.,  0.], [2.,  0.], [4.,  0.],
            [0.,  2.], [2.,  2.], [4.,  2.],
            [0.,  4.], [2.,  4.], [4.,  4.],
            [0.,  6.], [2.,  6.], [4.,  6.]
        ])
        dimension = 2
        actual = sub_sample_grid(grid, space=1, dimension=dimension)
        np.testing.assert_array_equal(actual, expected)

    def test_sub_sample_squared_grid_subspace_neq_1(self):
        grid = np.array([
            [0.,  0.], [1.,  0.], [2.,  0.], [3.,  0.], [4.,  0.], [5.,  0.], [6.,  0.],
            [0.,  1.], [1.,  1.], [2.,  1.], [3.,  1.], [4.,  1.], [5.,  1.], [6.,  1.],
            [0.,  2.], [1.,  2.], [2.,  2.], [3.,  2.], [4.,  2.], [5.,  2.], [6.,  2.],
            [0.,  3.], [1.,  3.], [2.,  3.], [3.,  3.], [4.,  3.], [5.,  3.], [6.,  3.],
            [0.,  4.], [1.,  4.], [2.,  4.], [3.,  4.], [4.,  4.], [5.,  4.], [6.,  4.],
            [0.,  5.], [1.,  5.], [2.,  5.], [3.,  5.], [4.,  5.], [5.,  5.], [6.,  5.],
            [0.,  6.], [1.,  6.], [2.,  6.], [3.,  6.], [4.,  6.], [5.,  6.], [6.,  6.]
        ])
        expected = np.array([
            [0.,  0.], [3.,  0.], [6.,  0.],
            [0.,  3.], [3.,  3.], [6.,  3.],
            [0.,  6.], [3.,  6.], [6.,  6.]
        ])
        dimension = 2
        actual = sub_sample_grid(grid, space=2, dimension=dimension)
        np.testing.assert_array_equal(actual, expected)

    def test_sub_sample_3d(self):
        grid = np.array([
            [0.,  0.,  0.], [0.,  0.,  1.], [0.,  0.,  2.], [0.,  0.,  3.], [0.,  0.,  4.],
            [1.,  0.,  0.], [1.,  0.,  1.], [1.,  0.,  2.], [1.,  0.,  3.], [1.,  0.,  4.],
            [2.,  0.,  0.], [2.,  0.,  1.], [2.,  0.,  2.], [2.,  0.,  3.], [2.,  0.,  4.],
            [3.,  0.,  0.], [3.,  0.,  1.], [3.,  0.,  2.], [3.,  0.,  3.], [3.,  0.,  4.],
            [4.,  0.,  0.], [4.,  0.,  1.], [4.,  0.,  2.], [4.,  0.,  3.], [4.,  0.,  4.],

            [0.,  1.,  0.],
            [0.,  1.,  1.],
            [0.,  1.,  2.],
            [0.,  1.,  3.],
            [0.,  1.,  4.],
            [1.,  1.,  0.], [1.,  1.,  1.], [1.,  1.,  2.], [1.,  1.,  3.], [1.,  1.,  4.],
            [2.,  1.,  0.], [2.,  1.,  1.], [2.,  1.,  2.], [2.,  1.,  3.], [2.,  1.,  4.],
            [3.,  1.,  0.], [3.,  1.,  1.], [3.,  1.,  2.], [3.,  1.,  3.], [3.,  1.,  4.],
            [4.,  1.,  0.], [4.,  1.,  1.], [4.,  1.,  2.], [4.,  1.,  3.], [4.,  1.,  4.],

            [0.,  2.,  0.], [0.,  2.,  1.], [0.,  2.,  2.], [0.,  2.,  3.], [0.,  2.,  4.],
            [1.,  2.,  0.], [1.,  2.,  1.], [1.,  2.,  2.], [1.,  2.,  3.], [1.,  2.,  4.],
            [2.,  2.,  0.], [2.,  2.,  1.], [2.,  2.,  2.], [2.,  2.,  3.], [2.,  2.,  4.],
            [3.,  2.,  0.], [3.,  2.,  1.], [3.,  2.,  2.], [3.,  2.,  3.], [3.,  2.,  4.],
            [4.,  2.,  0.], [4.,  2.,  1.], [4.,  2.,  2.], [4.,  2.,  3.], [4.,  2.,  4.],

            [0.,  3.,  0.], [0.,  3.,  1.], [0.,  3.,  2.], [0.,  3.,  3.], [0.,  3.,  4.],
            [1.,  3.,  0.], [1.,  3.,  1.], [1.,  3.,  2.], [1.,  3.,  3.], [1.,  3.,  4.],
            [2.,  3.,  0.], [2.,  3.,  1.], [2.,  3.,  2.], [2.,  3.,  3.], [2.,  3.,  4.],
            [3.,  3.,  0.], [3.,  3.,  1.], [3.,  3.,  2.], [3.,  3.,  3.], [3.,  3.,  4.],
            [4.,  3.,  0.], [4.,  3.,  1.], [4.,  3.,  2.], [4.,  3.,  3.], [4.,  3.,  4.],

            [0.,  4.,  0.], [0.,  4.,  1.], [0.,  4.,  2.], [0.,  4.,  3.], [0.,  4.,  4.],
            [1.,  4.,  0.], [1.,  4.,  1.], [1.,  4.,  2.], [1.,  4.,  3.], [1.,  4.,  4.],
            [2.,  4.,  0.], [2.,  4.,  1.], [2.,  4.,  2.], [2.,  4.,  3.], [2.,  4.,  4.],
            [3.,  4.,  0.], [3.,  4.,  1.], [3.,  4.,  2.], [3.,  4.,  3.], [3.,  4.,  4.],
            [4.,  4.,  0.], [4.,  4.,  1.], [4.,  4.,  2.], [4.,  4.,  3.], [4.,  4.,  4.]
        ])

        expected = np.array([
            [0.,  0.,  0.], [0.,  0.,  2.], [0.,  0.,  4.],
            [2.,  0.,  0.], [2.,  0.,  2.], [2.,  0.,  4.],
            [4.,  0.,  0.], [4.,  0.,  2.], [4.,  0.,  4.],

            [0.,  2.,  0.], [0.,  2.,  2.], [0.,  2.,  4.],
            [2.,  2.,  0.], [2.,  2.,  2.], [2.,  2.,  4.],
            [4.,  2.,  0.], [4.,  2.,  2.], [4.,  2.,  4.],

            [0.,  4.,  0.], [0.,  4.,  2.], [0.,  4.,  4.],
            [2.,  4.,  0.], [2.,  4.,  2.], [2.,  4.,  4.],
            [4.,  4.,  0.], [4.,  4.,  2.], [4.,  4.,  4.]
        ])
        actual = sub_sample_grid(grid, space=1)
        np.testing.assert_array_equal(actual, expected)
