from unittest import TestCase

import numpy as np

from kde.utils.eigenvaluesvectors import eigenValuesAndVectors, _eig_C, _eig_Python

class TestEig(TestCase):
    def test_eig_Python(self):
        self.eig_test_helper(lambda data: eigenValuesAndVectors(data, _eig_Python))

    def test_eig_C(self):
        self.eig_test_helper(lambda data: eigenValuesAndVectors(data, _eig_C))

    def eig_test_helper(self, the_function):
        data = np.diag((1.0, 2.0, 3.0))
        expected_values = np.array([1.0, 2.0, 3.0])
        expected_vectors = np.array([
            [1.0,  0.0,  0.0],
            [0.0,  1.0,  0.0],
            [0.0,  0.0,  1.0]
        ])


        actual_values, actual_vectors = the_function(data)

        np.testing.assert_array_almost_equal(expected_values, actual_values)
        np.testing.assert_array_almost_equal(expected_vectors, actual_vectors)


class EigImpAbstractTest(object):

    def setUp(self):
        super().setUp()
        self._implementation = None

    def test_eigenValuesAndVectors_0(self):
        data = np.diag((1.0, 2.0, 3.0))
        expected_values = np.array([1.0, 2.0, 3.0])
        expected_vectors = np.array([
            [1.0,  0.0,  0.0],
            [0.0,  1.0,  0.0],
            [0.0,  0.0,  1.0]
        ])

        actual_values, actual_vectors = self._implementation(data)

        np.testing.assert_array_almost_equal(expected_values, actual_values)
        np.testing.assert_array_almost_equal(expected_vectors, actual_vectors)


class Test_Eig_C(EigImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._implementation = _eig_C


class Test_Eig_Python(EigImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._implementation = _eig_Python