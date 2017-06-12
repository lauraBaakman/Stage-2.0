from unittest import TestCase
import warnings

import numpy as np

from kde.utils.knn import KNN, KNNException


class TestKNN(TestCase):

    def setUp(self):
        self._patterns = np.array([
            [0, 0],
            [1, 1],
            [2, 3],
            [4, 7]
        ], dtype=np.float64)
        self._k = 2
        self._expected = np.array([
            [0, 0],
            [1, 1]
        ], dtype=np.float64)
        self._expected.sort(axis=0)

    def test_find_k_nearest_neighbours_implicit_C(self):
        knn = KNN(patterns=self._patterns)
        pattern = np.array([1, 1], dtype=np.float64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.assertEqual(len(w), 0)
            actual = knn.find_k_nearest_neighbours(pattern=pattern, k=self._k)
            actual.sort(axis=0)
            np.testing.assert_array_almost_equal(actual, self._expected)

    def test_validate_k_1(self):
        knn = KNN(patterns=self._patterns)
        try:
            knn._validate_k(-2)
        except KNNException as e:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_validate_k_2(self):
        knn = KNN(patterns=self._patterns)
        try:
            knn._validate_k(20)
        except KNNException as e:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')