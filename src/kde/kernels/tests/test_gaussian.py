from unittest import TestCase, skip

import numpy as np

from kde.kernels.gaussian import Gaussian, _Gaussian_C, _Gaussian_Python


class TestGaussian(TestCase):

    def test_to_C_enum(self):
        actual = Gaussian(None, None).to_C_enum()
        expected = 3
        self.assertEqual(actual, expected)

    def test__validate_mean_covariance_combination_0(self):
        # valid combination
        self.fail()

    def test__validate_mean_covariance_combination_1(self):
        # Invalid dimension of mean compared to covariance matrix
        self.fail()

    def test__validate_eigen_values_pdf_combination_0(self):
        # Valid combination
        self.fail()

    def test__validate_eigen_values_pdf_combination_1(self):
        # Invalid number of eigenvalues compared to dimension of the mean
        self.fail()

    def test__validate_xs_pdf_combination_0(self):
        # valid combination
        self.fail()

    def test__validate_xs_pdf_combination_1(self):
        # Invalid dimension of xs, if single pattern
        self.fail()

    def test__validate_xs_pdf_combination_2(self):
        # Invalid dimension of xs, if multiple patterns
        self.fail()

    def test__validate_parameters_0(self):
        # Valid combination
        self.fail()

    def test__validate_parameters_1(self):
        # Invalid combination
        self.fail()

    def test__validate_mean_0(self):
        # valid mean
        self.fail()

    def test__validate_mean_1(self):
        # invalid mean
        self.fail()

    def test__validate_covariance_matrix_0(self):
        # Valid covariance matrix
        self.fail()

    def test__validate_covariance_matrix_1(self):
        # Invalid covariance matrix: not square
        self.fail()

    def test_evaluate_default_implementation(self):
        covariance_matrix = np.array([[1, 0], [0, 1]])
        mean = np.array([0, 0])
        pattern = np.array([0.5, 0.5])
        expected = 0.123949994309653
        actual = Gaussian(covariance_matrix, mean).evaluate(pattern)
        self.assertAlmostEqual(expected, actual)

    def test_scaling_factor(self):
        eigen_values = np.array([4.0, 9.0, 16.0, 25.0])
        h = 0.5
        expected = 0.5 * np.sqrt(7.5)
        actual = Gaussian(None, None).scaling_factor(general_bandwidth=h, eigen_values=eigen_values)
        self.assertAlmostEqual(expected, actual)


class GaussianImpAbstractTest(object):

    def test_evaluate_0(self):
        covariance_matrix = np.array([[1, 0], [0, 1]])
        mean = np.array([0, 0])
        pattern = np.array([0.5, 0.5])
        expected = 0.123949994309653
        actual = self._kernel_class(covariance_matrix, mean).evaluate(pattern)
        self.assertAlmostEqual(expected, actual)

    def test_evaluate_1(self):
        covariance_matrix = np.array([[0.5, 0.5], [0.5, 1.5]])
        mean = np.array([0, 0])
        pattern = np.array([0.5, 0.5])
        expected = 0.175291763008779
        actual = self._kernel_class(covariance_matrix, mean).evaluate(pattern)
        self.assertAlmostEqual(expected, actual)

    def test_evaluate_2(self):
        covariance_matrix = np.array([[1, 0], [0, 1]])
        mean = np.array([2, 2])
        pattern = np.array([0.5, 0.5])
        expected = 0.016774807587073
        actual = self._kernel_class(covariance_matrix, mean).evaluate(pattern)
        self.assertAlmostEqual(expected, actual)

    def test_evaluate_3(self):
        covariance_matrix = np.array([[0.5, 0.5], [0.5, 1.5]])
        mean = np.array([2, 2])
        pattern = np.array([0.5, 0.5])
        expected = 0.023723160395838
        actual = self._kernel_class(covariance_matrix, mean).evaluate(pattern)
        self.assertAlmostEqual(expected, actual)

    def test_scaling_factor(self):
        eigen_values = np.array([4.0, 9.0, 16.0, 25.0])
        h = 0.5
        expected = 0.5 * np.sqrt(7.5)
        actual = self._kernel_class(None, None).scaling_factor(general_bandwidth=h, eigen_values=eigen_values)
        self.assertAlmostEqual(expected, actual)


class Test_Gaussian_Python(GaussianImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._kernel_class = _Gaussian_Python

    def setUp(self):
        super().setUp()
        self._kernel_class = _Gaussian_Python


@skip("The C implementation of the Gaussian kernel has not yet been written.")
class Test_Gaussian_C(GaussianImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._kernel_class = _Gaussian_C