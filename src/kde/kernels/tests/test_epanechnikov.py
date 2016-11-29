from unittest import TestCase

import numpy as np

from kde.kernels import Epanechnikov
import kde.kernels.epanechnikov as epanechnikov


class TestEpanechnikov(TestCase):

    def setUp(self):
        super().setUp()
        self.data = {
            'u_1_1D': np.array([5]),
            'u_2_1D': np.array([0.5]),
            'u_3_1D': np.array([- 0.5]),
            'u_4_1D': np.array([[5], [0.5]]),

            'u_1_2D': np.array([5, 0.5]),
            'u_2_2D': np.array([0.5, 0.5]),
            'u_3_2D': np.array([[0.3, 0.5], [0.5, 0.5]]),
        }

    def test_evaluate_1D_1(self):
        kernel = Epanechnikov(dimension=1)
        actual = kernel.evaluate(self.data['u_1_1D'])
        expected = 0
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_1D_2(self):
        kernel = Epanechnikov(dimension=1)
        actual = kernel.evaluate(self.data['u_2_1D'])
        expected = 0.5625
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_1D_3(self):
        kernel = Epanechnikov(dimension=1)
        actual = kernel.evaluate(self.data['u_3_1D'])
        expected = 0.5625
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_1D_4(self):
        kernel = Epanechnikov(dimension=1)
        actual = kernel.evaluate(self.data['u_4_1D'])
        expected = np.array([-0, 0.5625])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_2D_1_mult(self):
        kernel = Epanechnikov(
            dimension=2,
            multivariate_approach=epanechnikov.Multivariate.multiplication)
        actual = kernel.evaluate(self.data['u_1_2D'])
        expected = 0
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_2D_2_mult(self):
        kernel = Epanechnikov(
            dimension=2,
            multivariate_approach=epanechnikov.Multivariate.multiplication)
        actual = kernel.evaluate(self.data['u_2_2D'])
        expected = 0.5625 * 0.5625
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_2D_3_mult(self):
        kernel = Epanechnikov(
            dimension=2,
            multivariate_approach=epanechnikov.Multivariate.multiplication)
        actual = kernel.evaluate(self.data['u_3_2D'])
        expected = np.array([0.383906250000000, 0.5625 * 0.5625])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_2D_1_norm(self):
        kernel = Epanechnikov(
            dimension=2,
            multivariate_approach=epanechnikov.Multivariate.norm)
        actual = kernel.evaluate(self.data['u_1_2D'])
        expected = 0.5625
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_2D_2_norm(self):
        kernel = Epanechnikov(
            dimension=2,
            multivariate_approach=epanechnikov.Multivariate.norm)
        actual = kernel.evaluate(self.data['u_2_2D'])
        expected = 0.795495128834866
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_2D_3_norm(self):
        kernel = Epanechnikov(
            dimension=2,
            multivariate_approach=epanechnikov.Multivariate.norm)
        actual = kernel.evaluate(self.data['u_3_2D'])
        expected = np.array([0.884427781110476, 0.795495128834866])
        np.testing.assert_array_almost_equal(actual, expected)