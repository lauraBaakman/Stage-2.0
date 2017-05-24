from unittest import TestCase

import numpy as np

from kde.utils.covariance import covariance, _covariance_C, _covariance_Python

class TestCovariance(TestCase):
    def test_covariance_Python_implicit(self):
        data = np.array([
            [1, 2],
            [1, 3],
            [4, 5],
            [3, 4]
        ])
        expected = np.array([
            [2.250000000000000, 1.833333333333333],
            [1.833333333333333, 1.666666666666667]
        ])
        actual = covariance(data)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_covariance_Python_explicit(self):
        data = np.array([
            [1, 2],
            [1, 3],
            [4, 5],
            [3, 4]
        ])
        expected = np.array([
            [2.250000000000000, 1.833333333333333],
            [1.833333333333333, 1.666666666666667]
        ])
        actual = covariance(data, implementation=_covariance_Python)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_covariance_C_explicit(self):
        data = np.array([
            [1, 2],
            [1, 3],
            [4, 5],
            [3, 4]
        ])
        expected = np.array([
            [2.250000000000000, 1.833333333333333],
            [1.833333333333333, 1.666666666666667]
        ])
        actual = covariance(data, implementation=_covariance_C)
        np.testing.assert_array_almost_equal(actual, expected)


class CovarianceImpAbstractTest(object):

    def setUp(self):
        super().setUp()
        self._implementation = None

    def test_covariance_0(self):
        data = np.array([
            [1, 2],
            [1, 3],
            [4, 5],
            [3, 4]
        ])
        expected = np.array([
            [2.250000000000000, 1.833333333333333],
            [1.833333333333333, 1.666666666666667]
        ])
        actual = self._implementation(data)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_covariance_1(self):
        data = np.array([
            [1, 2],
            [1, 3],
        ])
        expected = np.array([
            [0.0, 0.0],
            [0.0, 0.500000000000000]
        ])
        actual = self._implementation(data)
        np.testing.assert_array_almost_equal(actual, expected)


class Test_Covariance_C(CovarianceImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
        self._implementation = _covariance_C


class Test_Covariance_Python(CovarianceImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
        self._implementation = _covariance_Python
