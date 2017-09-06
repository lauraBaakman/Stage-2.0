from unittest import TestCase, skip

import numpy as np

from kde.utils.eigenvalues import eigenvalues, _eigenvalues_C, _eigenvalues_Python
import kde.utils.eigenvalues as ev


class TestEig(TestCase):
    def test_eig_Python_0(self):
        self.eig_test_helper_0(lambda data: eigenvalues(data, _eigenvalues_Python))

    def test_eig_C_0(self):
        self.eig_test_helper_0(lambda data: eigenvalues(data, _eigenvalues_C))

    def eig_test_helper_0(self, the_function):
        data = np.diag((1.0, 2.0, 3.0))
        expected_values = np.array([1.0, 2.0, 3.0])

        actual_values = the_function(data)

        np.testing.assert_array_almost_equal(expected_values, actual_values)

    def test_eig_Python_1(self):
        self.eig_test_helper_1(lambda data: eigenvalues(data, _eigenvalues_Python))

    def test_eig_C_1(self):
        self.eig_test_helper_1(lambda data: eigenvalues(data, _eigenvalues_C))

    def test_eig_Python_2(self):
        self.eig_test_helper_2(lambda data: eigenvalues(data, _eigenvalues_Python))

    def test_eig_C_2(self):
        self.eig_test_helper_2(lambda data: eigenvalues(data, _eigenvalues_C))

    def eig_test_helper_1(self, the_function):
        data = np.array([
            [1.0000, 0.5000, 0.3333, 0.2500],
            [0.5000, 0.3333, 0.2500, 0.2000],
            [0.3333, 0.2500, 0.2000, 0.1667],
            [0.2500, 0.2000, 0.1667, 0.1429],
        ])
        expected_values = sorted(np.array([0.000096702304023, 0.006738273605761, 0.169141220221450, 1.500214280059243]))

        actual_values = sorted(the_function(data))

        np.testing.assert_array_almost_equal(expected_values, actual_values, decimal=4)

    def eig_test_helper_2(self, the_function):
        data = np.array([
            [+0.081041292578536,  -0.003049670687501],
            [-0.003049670687501,  +0.083535264541089],
        ])
        expected_values = sorted(np.array([0.078993515239071, 0.085583041880554]))

        actual_values = sorted(the_function(data))

        np.testing.assert_array_almost_equal(expected_values, actual_values, decimal=4)


class EigImpAbstractTest(object):

    def setUp(self):
        super(EigImpAbstractTest, self).setUp()
        self._implementation = None

    def test_eigenValues_0(self):
        data = np.diag((1.0, 2.0, 3.0))
        expected_values = np.array([1.0, 2.0, 3.0])

        actual_values = self._implementation(data)

        np.testing.assert_array_almost_equal(expected_values, actual_values)

    def test_eigenValues_1(self):
        data = np.array([
            [0.0796, 0.0007, -0.0010],
            [0.0007, 0.0830, -0.0041],
            [- 0.0010, -0.0041, 0.0828],
        ])
        expected_values = sorted(np.array([0.078705439938896, 0.079497634589592, 0.087185641691132]))
        actual_values = sorted(self._implementation(data))

        np.testing.assert_array_almost_equal(expected_values, actual_values, 4)


class Test_Eig_C(EigImpAbstractTest, TestCase):

    def setUp(self):
        super(Test_Eig_C, self).setUp()
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
        super(Test_Eig_Python, self).setUp()
        self._implementation = _eigenvalues_Python