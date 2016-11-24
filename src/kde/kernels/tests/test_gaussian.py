from unittest import TestCase

import numpy as np

from kde.kernels import Gaussian


class TestGaussian(TestCase):

    def setUp(self):
        super().setUp()
        self.data = {
            'cov_1' : np.array([[1, 0], [0, 1]]),
            'cov_2': np.array([[0.5, 0.5], [0.5, 1.5]]),
            'mean_1': np.array([0, 0]),
            'mean_2': np.array([2, 2]),
            'pattern': np.array([0.5, 0.5]),
            'density_mean_1_cov_1': 0.123949994309653,
            'density_mean_1_cov_2': 0.175291763008779,
            'density_mean_2_cov_1': 0.016774807587073,
            'density_mean_2_cov_2': 0.023723160395838,
        }

    def test_center_get(self):
        kernel = Gaussian(
            mean=self.data('mean_1'),
            covariance_matrix=self.data('cov_1'))
        actual = kernel.center
        expected = self.data('mean_1')
        self.assertEqual(actual, expected)

    def test_center_set_1(self):
        kernel = Gaussian(
            mean=self.data('mean_1'),
            covariance_matrix=self.data('cov_1'))
        kernel.center= self.data('mean_2')
        actual_density = kernel._kernel.pdf(self.data('pattern'))
        expected_density = self.data('density_mean_2_cov_1')
        self.assertAlmostEqual(actual_density, expected_density)

    def test_center_set_2(self):
        kernel = Gaussian(
            covariance_matrix=self.data('cov_1'))
        kernel.center= self.data('mean_1')
        actual_density = kernel._kernel.pdf(self.data('pattern'))
        expected_density = self.data('density_mean_1_cov_1')
        self.assertAlmostEqual(actual_density, expected_density)

    def test_shape_get(self):
        kernel = Gaussian(
            mean=self.data('mean_1'),
            covariance_matrix=self.data('cov_1'))
        actual = kernel.shape
        expected = self.data('cov_1')
        self.assertEqual(actual, expected)

    def test_shape_set_1(self):
        kernel = Gaussian(
            mean=self.data('mean_1'),
            covariance_matrix=self.data('cov_1'))
        kernel.shape = self.data('cov_2')
        actual_density = kernel._kernel.pdf(self.data('pattern'))
        expected_density = self.data('density_mean_1_cov_2')
        self.assertAlmostEqual(actual_density, expected_density)

    def test_shape_set_2(self):
        kernel = Gaussian(
            mean=self.data('mean_1'))
        kernel.shape = self.data('cov_1')
        actual_density = kernel._kernel.pdf(self.data('pattern'))
        expected_density = self.data('density_mean_1_cov_1')
        self.assertAlmostEqual(actual_density, expected_density)

    def test_evaluate(self):
        kernel = Gaussian(
            mean=self.data('mean_1'),
            covariance_matrix=self.data('cov_1'))
        actual_density = kernel.evaluate(self.data('pattern'))
        expected_density = self.data('density_mean_1_cov_1')
        self.assertAlmostEqual(actual_density, expected_density)
