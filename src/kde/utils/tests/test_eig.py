from unittest import TestCase, skip

import numpy as np

from kde.utils.eigenvalues import eigenvalues, _eigenvalues_C, _eigenvalues_Python

class TestEig(TestCase):
    def test_eig_Python(self):
        self.eig_test_helper(lambda data: eigenvalues(data, _eigenvalues_Python))

    @skip("The C implementation of the computation of eigenvalues has not yet been written.")
    def test_eig_C(self):
        self.eig_test_helper(lambda data: eigenvalues(data, _eigenvalues_C))

    def eig_test_helper(self, the_function):
        data = np.diag((1.0, 2.0, 3.0))
        expected_values = np.array([1.0, 2.0, 3.0])

        actual_values = the_function(data)

        np.testing.assert_array_almost_equal(expected_values, actual_values)


class EigImpAbstractTest(object):

    def setUp(self):
        super().setUp()
        self._implementation = None

    def test_eigenValues_0(self):
        data = np.diag((1.0, 2.0, 3.0))
        expected_values = np.array([1.0, 2.0, 3.0])

        actual_values = self._implementation(data)

        np.testing.assert_array_almost_equal(expected_values, actual_values)

    def test_eigenValues_1(self):
        data = np.array([
            [0, 1],
            [-2, -3]
        ])
        expected_values = np.array([-1, -2])
        actual_values = self._implementation(data)

        np.testing.assert_array_almost_equal(expected_values, actual_values)

@skip("The C implementation of the computation of eigenvalues has not yet been written.")
class Test_Eig_C(EigImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._implementation = _eigenvalues_C


class Test_Eig_Python(EigImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._implementation = _eigenvalues_Python