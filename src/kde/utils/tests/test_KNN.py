from unittest import TestCase
import warnings
from unittest.test.testmock.support import examine_warnings

import numpy as np

from kde.utils.knn import _KNN_C, _KNN_Python, KNN, KNNException


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

    def test_find_k_nearest_neighbours_switch_to_Python(self):
        knn = KNN(patterns=self._patterns, implementation=_KNN_C)
        pattern = np.array([0.5, 0.5])
        expected = np.array([
            [0, 0],
            [1, 1]
        ], dtype=np.float64)
        expected.sort(axis=0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            actual = knn.find_k_nearest_neighbours(pattern=pattern, k=self._k)
            actual.sort(axis=0)
            self.assertEqual(len(w), 1)
            self.assertIsInstance(knn._implementation, _KNN_Python)
            np.testing.assert_array_almost_equal(actual, expected)


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


class KNNImpAbstractTest(object):

    def setUp(self):
        super().setUp()
        self.patterns = np.array([
            [0, 0],
            [1, 1],
            [2, 3],
            [4, 7]
        ], dtype=np.float64)
        self._implementation = None

    def test_find_k_nearest_neighbours_0(self):
        knn = self._implementation(self.patterns)
        pattern = np.array([0, 0])
        actual = knn.find_k_nearest_neighbours(pattern, k=2)
        expected = np.array([
            [0, 0],
            [1, 1]
        ], dtype=np.float64)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_find_k_nearest_neighbours_1(self):
        knn = self._implementation(self.patterns)
        pattern = np.array([1, 1])
        actual = knn.find_k_nearest_neighbours(pattern, k=2)
        expected = np.array([
            [0, 0],
            [1, 1]
        ], dtype=np.float64)
        expected.sort(axis=0)
        actual.sort(axis=0)
        np.testing.assert_array_almost_equal(actual, expected)


class Test_KNN_C(KNNImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
        self._implementation = _KNN_C

    def test_find_idx_of_pattern_0(self):
        knn = self._implementation(self.patterns)
        pattern = np.array([1, 1])
        expected = 1
        actual = knn._find_idx_of_pattern(pattern)
        self.assertEqual(actual, expected)

    def test_find_idx_of_pattern_1(self):
        knn = self._implementation(self.patterns)
        pattern = np.array([1, 0])
        try:
            knn._find_idx_of_pattern(pattern)
        except KNNException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_find_idx_of_pattern_2(self):
        patterns = np.array([
            [0, 0],
            [1, 1],
            [2, 3],
            [2, 3],
            [4, 7]
        ], dtype=np.float64)
        knn = self._implementation(patterns)
        pattern = np.array([2, 3])
        try:
            knn._find_idx_of_pattern(pattern)
        except KNNException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')


class Test_KNN_Python(KNNImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
        self._implementation = _KNN_Python
