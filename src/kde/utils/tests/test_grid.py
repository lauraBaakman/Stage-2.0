from unittest import TestCase

import numpy as np

from kde.utils.grid import Grid


class TestGrid(TestCase):
    def test___init__1D(self):
        range_1 = (0, 5)
        number_of_points = 6

        expected = np.array([[0], [1], [2], [3], [4], [5]])
        actual = Grid(range_1, number_of_grid_points=number_of_points).grid_points
        np.testing.assert_array_equal(expected, actual)

    def test___init__2D_int(self):
        range_1 = (0, 6)
        range_2 = (0, 2)
        number_of_points = 3

        expected = np.array([
            [0, 0],
            [3, 0],
            [6, 0],
            [0, 1],
            [3, 1],
            [6, 1],
            [0, 2],
            [3, 2],
            [6, 2],
        ])
        actual = Grid(range_1, range_2, number_of_grid_points=number_of_points).grid_points
        np.testing.assert_array_equal(expected, actual)

    def test_cover_with_padding_1d(self):
        points = np.array([[0.5], [1.1], [1.9], [3.2], [4.3], [5.4], [5.5]])
        padding = 0.5
        number_of_points = 3
        expected = np.array([[0], [3], [6]])
        actual = Grid.cover(points, padding=padding, number_of_grid_points=number_of_points).grid_points
        np.testing.assert_almost_equal(expected, actual)

    def test_cover_without_padding_1d(self):
        points = np.array([[0], [1.1], [1.9], [3.2], [4.3], [5.4], [6]])
        number_of_points = 3
        expected = np.array([[0], [3], [6]])
        actual = Grid.cover(points, number_of_grid_points=number_of_points).grid_points
        np.testing.assert_almost_equal(expected, actual)

    def test_cover_with_padding_2d(self):
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

    def test_cover_without_padding_2d(self):
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
