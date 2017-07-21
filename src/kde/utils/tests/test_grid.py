from unittest import TestCase

import numpy as np
import warnings

from kde.utils.grid import Grid


class TestGrid(TestCase):

    def test_cover_with_num_grid_points_with_padding_1d(self):
        points = np.array([[0.5], [1.1], [1.9], [3.2], [4.3], [5.4], [5.5]])
        padding = 0.5
        number_of_points = 3
        expected = np.array([[0], [3], [6]])
        actual = Grid.cover(points, padding=padding, number_of_grid_points=number_of_points).grid_points
        np.testing.assert_almost_equal(expected, actual)

    def test_cover_with_num_grid_points_without_padding_1d(self):
        points = np.array([[0], [1.1], [1.9], [3.2], [4.3], [5.4], [6]])
        number_of_points = 3
        expected = np.array([[0], [3], [6]])
        actual = Grid.cover(points, number_of_grid_points=number_of_points).grid_points
        np.testing.assert_almost_equal(expected, actual)

    def test_cover_with_num_grid_points_with_padding_2d(self):
        points = np.array([
            [0.3, 0.5],
            [0.9, 0.4],
            [0.4, 0.1],
            [1.0, 0.2],
            [0.4, 0.4],
            [0.6, 0.3]
        ])
        padding = 0.2
        number_of_points = 2
        expected = np.array([
            [0.1, -0.1],
            [1.2, -0.1],
            [0.1, 0.7],
            [1.2, 0.7]
        ])
        actual = Grid.cover(points, padding=padding, number_of_grid_points=number_of_points).grid_points
        np.testing.assert_almost_equal(expected, actual)

    def test_cover_with_num_grid_points_without_padding_2d(self):
        points = np.array([
            [0.3, 0.5],
            [0.9, 0.4],
            [0.4, 0.1],
            [1.0, 0.2],
            [0.4, 0.4],
            [0.6, 0.3]
        ])
        number_of_points = 2
        expected = np.array([
            [0.3, 0.1],
            [1.0, 0.1],
            [0.3, 0.5],
            [1.0, 0.5]
        ])
        actual = Grid.cover(points, number_of_grid_points=number_of_points).grid_points
        np.testing.assert_almost_equal(expected, actual)

    def test_cover_with_cellsize_without_padding_1d(self):
        points = np.array([[-17], [1.1], [1.9], [3.2], [4.3], [5.4], [34.0]])
        cell_size = 5
        expected = np.array([[-19], [-14], [-9], [-4], [1], [6], [11], [16], [21], [26], [31], [36]])
        actual = Grid.cover(points, cell_size=cell_size).grid_points
        np.testing.assert_almost_equal(expected, actual)

    def test_cover_with_cellsize_without_padding_infinite_no_num_grid_points(self):
        with warnings.catch_warnings(record=True):
            points = np.array([[0], [1.1], [1.9], [3.2], [4.3], [5.4], [95]])
            cell_size = float('inf')
            expected = np.array([
                [00], [05], [10], [15], [20], [25], [30],
                [35], [40], [45], [50], [55], [60], [65],
                [70], [75], [80], [85], [90], [95]]
            )
            actual = Grid.cover(points, cell_size=cell_size).grid_points
            np.testing.assert_almost_equal(expected, actual)

    def test_cover_with_cellsize_without_padding_infinite_num_grid_points(self):
        with warnings.catch_warnings(record=True):
            points = np.array([[0], [1.1], [1.9], [3.2], [4.3], [5.4], [6]])
            number_of_points = 3
            cell_size = float('inf')
            expected = np.array([[0], [3], [6]])
            actual = Grid.cover(points, cell_size=cell_size, number_of_grid_points=number_of_points).grid_points
            np.testing.assert_almost_equal(expected, actual)

    def test_cover_with_cellsize_without_padding_1d_floats(self):
        points = np.array([[-7.3], [-5.5], [8.2]])
        cell_size = 4.5
        expected = np.array([[-8.55], [-4.05], [0.45], [4.95], [9.45]])
        actual = Grid.cover(points, cell_size=cell_size).grid_points
        np.testing.assert_almost_equal(expected, actual)

    def test_cover_with_cellsize_remainder_zero(self):
        points = np.array([[-2.0], [-1.5], [2.0]])
        cell_size = 2
        expected = np.array([[-2.0], [0.0], [2.0]])
        actual = Grid.cover(points, cell_size=cell_size).grid_points
        np.testing.assert_almost_equal(expected, actual)

    def test_cover_with_cellsize_without_padding_2d(self):
        points = np.array([
            [-7.0, 3.00],
            [+0.9, 7.26],
            [+0.4, 3.41],
            [+1.0, 8.50],
            [+0.4, 9.00],
            [+6.0, 11.0]
        ])
        cell_size = 5
        expected = np.array([
            [-8, 2],
            [-3, 2],
            [+2, 2],
            [+7, 2],
            [-8, 7],
            [-3, 7],
            [+2, 7],
            [+7, 7],
            [-8, 12],
            [-3, 12],
            [+2, 12],
            [+7, 12]
        ])
        actual = Grid.cover(points, cell_size=cell_size).grid_points
        np.testing.assert_almost_equal(expected, actual)

    def test_array_flags_array_order(self):
        points = np.array([[-2.0], [-1.5], [2.0]])
        cell_size = 2
        expected = np.array([[-2.0], [0.0], [2.0]])
        actual = Grid.cover(points, cell_size=cell_size).grid_points
        self.assertTrue(actual.flags['C_CONTIGUOUS'])
        self.assertFalse(actual.flags['F_CONTIGUOUS'])

    def test_array_flags_array_owndata(self):
        points = np.array([[-2.0], [-1.5], [2.0]])
        cell_size = 2
        expected = np.array([[-2.0], [0.0], [2.0]])
        actual = Grid.cover(points, cell_size=cell_size).grid_points
        self.assertTrue(actual.flags['OWNDATA'])
