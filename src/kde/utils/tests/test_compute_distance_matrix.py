from unittest import TestCase

import numpy as np

from kde.utils.distanceMatrix import compute_distance_matrix, _compute_distance_matrix_C, _compute_distance_matrix_Python

class TestCompute_distance_matrix(TestCase):


    def setUp(self):
        super().setUp()
        self.patterns = np.array([
            [0, 0],
            [1, 1],
            [2, 3],
            [4, 7]
        ])
        self.expected = np.matrix([
            [0, 2, 12, 65],
            [2, 0, 5, 45],
            [13, 5, 0, 20],
            [65, 45, 20, 0]
        ])

    def test_compute_distance_matrix_C_implicit(self):
        actual = compute_distance_matrix(self.patterns)
        np.testing.assert_array_equal(actual, self.expected)

    def test_compute_distance_matrix_C_explicit(self):
        actual = compute_distance_matrix(self.patterns, implementation=_compute_distance_matrix_C)
        np.testing.assert_array_equal(actual, self.expected)

    def test_compute_distance_matrix_C_python(self):
        actual = compute_distance_matrix(self.patterns, implementation=_compute_distance_matrix_Python)
        np.testing.assert_array_equal(actual, self.expected)


class Compute_distance_matrixAbstractTest(object):

    def setUp(self):
        super().setUp()
        self._implementation = None

    def test_compute_distance(self):
        patterns = np.array([
            [0, 0],
            [1, 1],
            [2, 3],
            [4, 7]
        ])
        expected = np.matrix([
            [0, 2, 12, 65],
            [2, 0, 5, 45],
            [13, 5, 0, 20],
            [65, 45, 20, 0]
        ])
        actual = self._implementation(patterns)
        np.testing.assert_array_equal(actual, expected)


class TestCompute_distance_matrix_C(Compute_distance_matrixAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._implementation = _compute_distance_matrix_C


class TestCompute_distance_matrix_Python(Compute_distance_matrixAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._implementation = _compute_distance_matrix_Python
