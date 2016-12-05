from unittest import TestCase

import numpy as np

from kdeUtils.grid import Grid


class TestGrid(TestCase):

    def test___init__1D(self):
        range_1 = (0, 5)
        number_of_points = 6

        expected = np.array([[0], [1], [2], [3], [4], [5]])
        actual = Grid(number_of_points, range_1).grid_points
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
        actual = Grid(number_of_points, range_1, range_2).grid_points
        np.testing.assert_array_equal(expected, actual)