from unittest import TestCase

import numpy as np

from kde.utils.distanceMatrix import compute_distance_matrix

class TestCompute_distance_matrix(TestCase):
    def setUp(self):
        super().setUp()
        self.patterns = np.array([
            [0, 0],
            [1, 1],
            [2, 3],
            [4, 7]
        ])
        self.expected = np.array([
            [0, 2, 13, 65],
            [2, 0, 5, 45],
            [13, 5, 0, 20],
            [65, 45, 20, 0]
        ], dtype=np.float64)

    def test_compute_distance_matrix(self):
        actual = compute_distance_matrix(self.patterns)
        np.testing.assert_array_almost_equal(actual, self.expected)