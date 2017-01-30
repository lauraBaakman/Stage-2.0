from unittest import TestCase

import numpy as np

from kde.datavalidation import EstimatorDataValidator, InvalidEstimatorArguments


class TestEstimatorDataValidator(TestCase):

    def test__array_has_two_dimensions_1(self):
        x_s = np.array([[0.5, 0.5], [0.1, 0.1]])

        actual = EstimatorDataValidator._array_has_two_dimensions(x_s)
        self.assertIsNone(actual)

    def test__array_has_two_dimensions_2(self):
        x_s = np.array([
            [[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]],
            [[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]]])

        try:
            EstimatorDataValidator._array_has_two_dimensions(x_s)
        except InvalidEstimatorArguments:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__array_has_two_dimensions_3(self):
        x_s = np.array([
            [[0.5, 0.5], [0.1, 0.1]],
            [[0.5, 0.5], [0.1, 0.1]]])

        try:
            EstimatorDataValidator._array_has_two_dimensions(x_s)
        except InvalidEstimatorArguments:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__do_elements_have_same_dimension_1(self):
        xi_s = np.array([[0.5, 0.5], [0.1, 0.1]])
        x_s = np.array([[0.5, 0.5], [0.1, 0.1], [0.5, 0.5], [0.1, 0.1]])

        actual = EstimatorDataValidator._do_elements_have_same_dimension(xi_s, x_s)
        self.assertIsNone(actual)

    def _do_elements_have_same_dimension_helper(self, *args):
        try:
            EstimatorDataValidator._do_elements_have_same_dimension(*args)
        except InvalidEstimatorArguments:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__do_elements_have_same_dimension_2(self):
        xi_s = np.array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
        x_s = np.array([[0.5, 0.5], [0.1, 0.1], [0.5, 0.5], [0.1, 0.1]])
        self._do_elements_have_same_dimension_helper(xi_s, x_s)

    def test__do_elements_have_same_dimension_3(self):
        xi_s = np.array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
        x_s = np.array([[0.5, 0.5], [0.1, 0.1]])
        self._do_elements_have_same_dimension_helper(xi_s, x_s)

    def test__do_elements_have_same_dimension_4(self):
        xi_s = np.array([[0.5, 0.5], [0.1, 0.1]])
        x_s = np.array([[0.5, 0.5], [0.1, 0.1], [0.5, 0.5], [0.1, 0.1]])
        x2_s = np.array([[0.5, 0.5], [0.1, 0.1], [0.5, 0.5], [0.1, 0.1], [0.3, 0.3]])

        actual = EstimatorDataValidator._do_elements_have_same_dimension(xi_s, x_s, x2_s)
        self.assertIsNone(actual)

    def test__do_elements_have_same_dimension_5(self):
        xi_s = np.array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
        x2_s = np.array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
        x_s = np.array([[0.5, 0.5], [0.1, 0.1]])
        self._do_elements_have_same_dimension_helper(xi_s, x2_s, x_s, )

    def test__do_elements_have_same_dimension_6(self):
        xi_s = np.array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
        x2_s = np.array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
        x_s = np.array([[0.5, 0.5], [0.1, 0.1]])
        self._do_elements_have_same_dimension_helper(xi_s, x2_s, x_s)