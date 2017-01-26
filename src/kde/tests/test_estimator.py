from unittest import TestCase

import numpy as np

from kde.estimator import Estimator, InvalidEstimatorArguments


class TestEstimator(TestCase):
    def test__validate_data_1(self):

        xi_s = np.array([[0.5, 0.5], [0.1, 0.1]])
        x_s = np.array([[0.5, 0.5], [0.1, 0.1], [0.5, 0.5], [0.1, 0.1]])

        actual = Estimator()._validate_data(xi_s, x_s)
        self.assertIsNone(actual)

    def test__have_same_dimension_1(self):
        xi_s = np.array([[0.5, 0.5], [0.1, 0.1]])
        x_s = np.array([[0.5, 0.5], [0.1, 0.1], [0.5, 0.5], [0.1, 0.1]])

        actual = Estimator()._have_same_dimension(xi_s, x_s)
        self.assertIsNone(actual)

    def test__have_same_dimension_2(self):
        xi_s = np.array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
        x_s = np.array([[0.5, 0.5], [0.1, 0.1], [0.5, 0.5], [0.1, 0.1]])
        try:
            Estimator()._have_same_dimension(x_s, xi_s)
        except InvalidEstimatorArguments:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__have_same_dimension_3(self):
        xi_s = np.array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
        x_s = np.array([[0.5, 0.5], [0.1, 0.1]])

        try:
            Estimator()._have_same_dimension(x_s, xi_s)
        except InvalidEstimatorArguments:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__array_has_correct_shape_1(self):
        x_s = np.array([[0.5, 0.5], [0.1, 0.1]])

        actual = Estimator()._array_has_correct_shape(x_s)
        self.assertIsNone(actual)

    def test__array_has_correct_shape_2(self):
        x_s = np.array([
            [[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]],
            [[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]]])

        try:
            Estimator()._array_has_correct_shape(x_s)
        except InvalidEstimatorArguments:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__array_has_correct_shape_3(self):
        x_s = np.array([
            [[0.5, 0.5], [0.1, 0.1]],
            [[0.5, 0.5], [0.1, 0.1]]])

        try:
            Estimator()._array_has_correct_shape(x_s)
        except InvalidEstimatorArguments:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')