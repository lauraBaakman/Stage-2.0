from unittest import TestCase, skip

import numpy as np

from kde.utils.eigenvalues import eigenvalues, _eigenvalues_C, _eigenvalues_Python
import kde.utils.eigenvalues as ev


class TestEig(TestCase):
    def test_eig_Python(self):
        self.eig_test_helper(lambda data: eigenvalues(data, _eigenvalues_Python))

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
            [0.0, 1.0],
            [-2.0, -3.0]
        ])
        expected_values = np.array([-1, -2])
        actual_values = self._implementation(data)

        np.testing.assert_array_almost_equal(expected_values, actual_values)


class Test_Eig_C(EigImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._implementation = _eigenvalues_C

    def test__is_square_1(self):
        data = np.array([[2, 2], [3, 3]])
        actual = ev._is_square(data)
        self.assertIsNone(actual)

    def test__is_square_2(self):
        data = np.array([[2, 2, 2], [3, 3, 3]])
        try:
            ev._is_square(data)
        except ValueError:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__is_square_3(self):
        data = np.array([[2, 2], [3, 3], [4, 4]])
        try:
            ev._is_square(data)
        except ValueError:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__has_two_dimensions_1(self):
        data = np.array([[2, 2], [3, 3]])
        actual = ev._has_two_dimensions(data)
        self.assertIsNone(actual)

    def test__has_two_dimensions_2(self):
        data = np.array([1])
        try:
            ev._has_two_dimensions(data)
        except ValueError:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__has_two_dimensions_3(self):
        data = np.array([
            [[2, 2], [3, 3]],
            [[2, 2], [3, 3]]
        ])
        try:
            ev._has_two_dimensions(data)
        except ValueError:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__is_at_least_2_times_2_1(self):
        data = np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
        actual = ev._order_greater_than_two(data)
        self.assertIsNone(actual)

    def test__is_at_least_2_times_2_2(self):
        data = np.array([1])
        try:
            ev._order_greater_than_two(data)
        except ValueError:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__is_at_least_2_times_2_3(self):
        data = np.array([[1]])
        try:
            ev._order_greater_than_two(data)
        except ValueError:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__is_at_least_2_times_2_4(self):
        data = np.array([[1, 1], [2, 2]])
        actual = ev._order_greater_than_two(data)
        self.assertIsNone(actual)

    def test__validate_input_matrix_1(self):
        data = np.array([[1, 1], [2, 2]])
        actual = ev._validate_input_matrix(data)
        self.assertIsNone(actual)

    def test__validate_input_matrix_2(self):
        data = np.array([[1]])
        try:
            ev._validate_input_matrix(data)
        except ValueError:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

class Test_Eig_Python(EigImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._implementation = _eigenvalues_Python