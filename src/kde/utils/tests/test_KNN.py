from unittest import TestCase

import numpy as np

from kde.utils.knn import _KNN_C, _KNN_Python, KNN


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
        self._pattern = np.array([0.4, 0.4], dtype=np.float64)


    def test_find_k_nearest_neighbours_implicit_C(self):
        knn = KNN(patterns=self._patterns)
        actual = knn.find_k_nearest_neighbours(pattern=self._pattern, k=self._k)
        np.testing.assert_array_almost_equal   (actual, self._expected)

    def test_find_k_nearest_neighbours_explicit_C(self):
        knn = KNN(patterns=self._patterns, implementation=_KNN_C)
        actual = knn.find_k_nearest_neighbours(pattern=self._pattern, k=self._k)
        np.testing.assert_array_almost_equal(actual, self._expected)

    def test_find_k_nearest_neighbours_explicit_Python(self):
        knn = KNN(patterns=self._patterns, implementation=_KNN_Python)
        actual = knn.find_k_nearest_neighbours(pattern=self._pattern, k=self._k)
        np.testing.assert_array_almost_equal(actual, self._expected)

    def test_find_k_nearest_neighbours_1(self):
        knn = KNN(patterns=self._patterns)
        try:
            knn.find_k_nearest_neighbours(pattern=self._pattern, k=-2)
        except TypeError as e:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_find_k_nearest_neighbours_2(self):
        knn = KNN(patterns=self._patterns)
        try:
            knn.find_k_nearest_neighbours(pattern=self._pattern, k=20)
        except TypeError as e:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')


class KNNImpAbstractTest(object):

    def setUp(self):
        super().setUp()
        self.patterns = np.array([
            [0, 0],
            [1, 1],
            [2, 3],
            [4, 7]
        ], dtype=np.float64)
        self.pattern = np.array([0.4, 0.4], dtype=np.float64)
        self._implementation = None

    def test_find_k_nearest_neighbours_0(self):
        knn = self._implementation(self.patterns)
        actual = knn.find_k_nearest_neighbours(self.pattern, k=2)
        expected = np.array([
            [0, 0],
            [1, 1]
        ], dtype=np.float64)
        np.testing.assert_array_almost_equal(actual, expected)


class Test_KNN_C(KNNImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
        self._implementation = _KNN_C

    def _find_idx_of_pattern(self):
        knn = self._implementation(self.patterns)
        pattern = np.array([1, 1])
        expected = 1
        actual = knn._find_idx_of_pattern(pattern)
        self.assertEqual(actual, expected)


class Test_KNN_Python(KNNImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
        self._implementation = _KNN_Python
