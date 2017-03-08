from sys import implementation
from unittest import TestCase, skip

import numpy as np

from kde.kernels.gaussian import Gaussian, _Gaussian_C, _Gaussian_Python, _Gaussian
from kde.kernels.kernel import KernelException


class TestGaussian(TestCase):

    def test_to_C_enum(self):
        actual = Gaussian.to_C_enum()
        expected = 3
        self.assertEqual(actual, expected)

    def test_default_implementation(self):
        covariance_matrix = np.array([[1, 0], [0, 1]])
        mean = np.array([0, 0])
        pattern = np.array([0.5, 0.5])
        expected = 0.123949994309653
        actual = Gaussian(mean, covariance_matrix).evaluate(pattern)
        self.assertAlmostEqual(expected, actual)

    def test_alternative_implementation(self):
        covariance_matrix = np.array([[1, 0], [0, 1]])
        mean = np.array([0, 0])
        pattern = np.array([0.5, 0.5])
        expected = 0.123949994309653
        actual = Gaussian(mean, covariance_matrix, _Gaussian_Python).evaluate(pattern)
        self.assertAlmostEqual(expected, actual)

    def test_input_validation_alternative_implementation(self):
        covariance_matrix = np.array([[1, 0], [0, 1]])
        mean = np.array([0, 0, 0.5])
        try:
            Gaussian(mean, covariance_matrix, _Gaussian_Python)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_input_validation_default_implementation(self):
        covariance_matrix = np.array([[1, 0], [0, 1]])
        mean = np.array([0, 0, 0.5])
        try:
            Gaussian(mean, covariance_matrix)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    @skip("The C implementation of the Gaussian kernel has not yet been written.")
    def test_scaling_factor_default_implementation(self):
        eigen_values = np.array([4.0, 9.0, 16.0, 25.0])
        covariance_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        mean = np.array([2, 2, 3, 4])
        h = 0.5
        expected = 0.075534384933919
        kernel = Gaussian(mean, covariance_matrix, implementation=_Gaussian_C)
        actual = kernel.scaling_factor(general_bandwidth=h, eigen_values=eigen_values)
        self.assertAlmostEqual(expected, actual)

    def test_scaling_factor_alternative_implementation(self):
        eigen_values = np.array([4.0, 9.0, 16.0, 25.0])
        covariance_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        mean = np.array([2, 2, 3, 4])
        h = 0.5
        expected = 0.075534384933919
        kernel = Gaussian(mean, covariance_matrix, implementation=_Gaussian_Python)
        actual = kernel.scaling_factor(general_bandwidth=h, eigen_values=eigen_values)
        self.assertAlmostEqual(expected, actual)

class Test_Gaussian(TestCase):

    def setUp(self):
        def mock_init(self, mean, covariance_matrix):
            self._mean = mean
            self._covariance_matrix = covariance_matrix

        super().setUp()
        _Gaussian.__init__ = mock_init

    def test_dimension(self):
        mean = np.array([0.5, 0.5])
        covariance_matrix = np.array([
            [1.2, 2.1],
            [2.3, 3.2]
        ])
        actual = _Gaussian(mean, covariance_matrix).dimension
        expected = 2
        self.assertEqual(actual, expected)

    def test__validate_mean_covariance_combination_0(self):
        # valid combination
        mean = np.array([0.5, 0.5])
        covariance_matrix = np.array([
            [1.2, 2.1],
            [2.3, 3.2]
        ])
        actual = _Gaussian(None, None)._validate_mean_covariance_combination(mean, covariance_matrix)
        self.assertIsNone(actual)

    def test__validate_mean_covariance_combination_1(self):
        # Invalid dimension of mean compared to covariance matrix
        mean = np.array([0.5, 0.5, 0.5])
        covariance_matrix = np.array([
            [1.2, 2.1],
            [2.3, 3.2]
        ])
        try:
            _Gaussian(None, None)._validate_mean_covariance_combination(mean, covariance_matrix)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__validate_xs_pdf_combination_0(self):
        # valid combination 1D
        kernel = _Gaussian(
            mean=np.array([0.5, 0.5]),
            covariance_matrix=np.array([
                [1.2, 2.1],
                [2.3, 3.2]
            ])
        )
        xs = np.array([0.1, 0.2])
        actual = kernel._validate_xs_pdf_combination(xs)
        self.assertIsNone(actual)

    def test__validate_xs_pdf_combination_1(self):
        # valid combination ND
        kernel = _Gaussian(
            mean=np.array([0.5, 0.5]),
            covariance_matrix=np.array([
                [1.2, 2.1],
                [2.3, 3.2]
            ])
        )
        xs = np.array([[0.1, 0.2], [0.1, 0.2]])
        actual = kernel._validate_xs_pdf_combination(xs)
        self.assertIsNone(actual)

    def test__validate_xs_pdf_combination_2(self):
        # Invalid dimension of xs, if single pattern
        kernel = _Gaussian(
            mean=np.array([0.5, 0.5]),
            covariance_matrix=np.array([
                [1.2, 2.1],
                [2.3, 3.2]
            ])
        )
        xs = np.array([0.1, 0.2, 0.3])
        try:
            kernel._validate_xs_pdf_combination(xs)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__validate_xs_pdf_combination_3(self):
        # Invalid dimension of xs, if multiple patterns
        kernel = _Gaussian(
            mean=np.array([0.5, 0.5]),
            covariance_matrix=np.array([
                [1.2, 2.1],
                [2.3, 3.2]
            ])
        )
        xs = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
        try:
            kernel._validate_xs_pdf_combination(xs)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__validate_parameters_0(self):
        # Valid combination
        mean = np.array([0.5, 0.5])
        covariance_matrix = np.array([
            [1.2, 2.1],
            [2.3, 3.2]
        ])
        actual = _Gaussian(None, None)._validate_parameters(mean, covariance_matrix)
        self.assertIsNone(actual)

    def test__validate_parameters_1(self):
        # Invalid combination
        mean = np.array([0.5, 0.5, 0.6])
        covariance_matrix = np.array([
            [1.2, 2.1],
            [2.3, 3.2]
        ])
        try:
            _Gaussian(None, None)._validate_parameters(mean, covariance_matrix)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__validate_mean_0(self):
        # valid mean
        mean = np.array([0.5, 0.5, 0.6])
        actual = _Gaussian(None, None)._validate_mean(mean)
        self.assertIsNone(actual)

    def test__validate_mean_1(self):
        # invalid mean
        mean = np.array([
            [1.2, 2.1],
            [2.3, 3.2]
        ])
        try:
            _Gaussian(None, None)._validate_mean(mean)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__validate_covariance_matrix_0(self):
        # Valid covariance matrix
        covariance = np.array([
            [1.2, 2.1],
            [2.3, 3.2]
        ])
        actual = _Gaussian(None, None)._validate_covariance_matrix(covariance)
        self.assertIsNone(actual)

    def test__validate_covariance_matrix_1(self):
        # Invalid covariance matrix: not square
        covariance = np.array([
            [1.2, 2.1, 3.0],
            [2.3, 3.2, 4.0]
        ])
        try:
            _Gaussian(None, None)._validate_covariance_matrix(covariance)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test__validate_covariance_matrix_2(self):
        # Invalid covariance matrix: not 2D
        covariance = np.array([
            [
                [1.2, 2.1],
                [2.3, 3.2]
            ],
            [
                [1.2, 2.1],
                [2.3, 3.2]
            ]
        ])
        try:
            _Gaussian(None, None)._validate_covariance_matrix(covariance)
        except KernelException:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e))
        else:
            self.fail('ExpectedException not raised')

    def test_evaluate_C_implementation(self):
        covariance_matrix = np.array([[1, 0], [0, 1]])
        mean = np.array([0, 0])
        pattern = np.array([0.5, 0.5])
        expected = 0.123949994309653
        actual = Gaussian(mean, covariance_matrix, implementation=_Gaussian_C).evaluate(pattern)
        self.assertAlmostEqual(expected, actual)

    def test_evaluate_python_implementation(self):
        covariance_matrix = np.array([[1, 0], [0, 1]])
        mean = np.array([0, 0])
        pattern = np.array([0.5, 0.5])
        expected = 0.123949994309653
        actual = Gaussian(mean, covariance_matrix, implementation=_Gaussian_Python).evaluate(pattern)
        self.assertAlmostEqual(expected, actual)

    @skip("The C implementation of the scaling of the Gaussian kernel has not yet been written.")
    def test_scaling_factor_default_implementation(self):
        eigen_values = np.array([4.0, 9.0, 16.0, 25.0])
        covariance_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        h = 0.5
        expected = 0.075534384933919
        actual = Gaussian(None, covariance_matrix).scaling_factor(general_bandwidth=h, eigen_values=eigen_values)
        self.assertAlmostEqual(expected, actual)

    def test_scaling_factor_alternative_implementation(self):
        eigen_values = np.array([4.0, 9.0, 16.0, 25.0])
        covariance_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        h = 0.5
        expected = 0.075534384933919
        kernel = Gaussian(None, covariance_matrix, implementation=_Gaussian_Python)
        actual = kernel.scaling_factor(general_bandwidth=h, eigen_values=eigen_values)
        self.assertAlmostEqual(expected, actual)


class GaussianImpAbstractTest(object):

    def test_evaluate_0(self):
        covariance_matrix = np.array([[1, 0], [0, 1]])
        mean = np.array([0, 0])
        pattern = np.array([0.5, 0.5])
        expected = 0.123949994309653
        actual = self._kernel_class(mean, covariance_matrix).evaluate(pattern)
        self.assertAlmostEqual(expected, actual)

    def test_evaluate_1(self):
        covariance_matrix = np.array([[0.5, 0.5], [0.5, 1.5]])
        mean = np.array([0, 0])
        pattern = np.array([0.5, 0.5])
        expected = 0.175291763008779
        actual = self._kernel_class(mean, covariance_matrix).evaluate(pattern)
        self.assertAlmostEqual(expected, actual)

    def test_evaluate_2(self):
        covariance_matrix = np.array([[1, 0], [0, 1]])
        mean = np.array([2, 2])
        pattern = np.array([0.5, 0.5])
        expected = 0.016774807587073
        actual = self._kernel_class(mean, covariance_matrix).evaluate(pattern)
        self.assertAlmostEqual(expected, actual)

    def test_evaluate_3(self):
        covariance_matrix = np.array([[0.5, 0.5], [0.5, 1.5]])
        mean = np.array([2, 2])
        pattern = np.array([0.5, 0.5])
        expected = 0.023723160395838
        actual = self._kernel_class(mean, covariance_matrix).evaluate(pattern)
        self.assertAlmostEqual(expected, actual)

    def test_evaluate_4(self):
        covariance_matrix = np.array([[0.26358307, -0.13179154], [-0.13179154,  0.26358307]])
        mean = np.array([0, 0])
        pattern = np.array([0, 0])
        expected = 0.697223462789203
        actual = self._kernel_class(mean, covariance_matrix).evaluate(pattern)
        self.assertAlmostEqual(expected, actual)

    def test_evaluate_5(self):
        covariance_matrix = np.array([[0.5, 0.5], [0.5, 1.5]])
        mean = np.array([1, 0.5])
        pattern = np.array([[0, 0], [1, 1.5], [-0.5, 1.5]])
        expected = np.array([0.073072478360846, 0.136517362297204,  0.001042322923649])
        actual = self._kernel_class(mean, covariance_matrix).evaluate(pattern)
        np.testing.assert_array_almost_equal(actual, expected)


    def test_scaling_factor(self):
        eigen_values = np.array([4.0, 9.0])
        h = 0.5
        covariance_matrix = np.array([[0.5, 0.5], [0.5, 1.5]])
        mean = np.array([2, 2])
        expected = 0.102062072615966
        kernel = self._kernel_class(mean, covariance_matrix)
        actual = kernel.scaling_factor(general_bandwidth=h, eigen_values=eigen_values)
        self.assertAlmostEqual(expected, actual)


class Test_Gaussian_Python(GaussianImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._kernel_class = _Gaussian_Python

    def setUp(self):
        super().setUp()
        self._kernel_class = _Gaussian_Python


class Test_Gaussian_C(GaussianImpAbstractTest, TestCase):

    def setUp(self):
        super().setUp()
        self._kernel_class = _Gaussian_C

    def test_scaling_factor(self):
        self.skipTest("The C implementation of the scaling of the Gaussian kernel has not yet been written.")