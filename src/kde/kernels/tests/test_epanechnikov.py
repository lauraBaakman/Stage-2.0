from unittest import TestCase

import numpy as np

from kde.kernels import Epanechnikov
import kde.kernels.epanechnikov as epanechnikov


class TestEpanechnikov(TestCase):

    def setUp(self):
        super().setUp()
        self.data = {
            'u_1_1D': np.array([-5]),
            'u_2_1D': np.array([5]),
            'u_1_2D': np.array([5, 3]),
            'u_2_2D': np.array([-5, 3]),
        }

    def test_evaluate_1D_1(self):
        kernel = Epanechnikov(dimension=1)
        actual = kernel.evaluate(self.data['u_1_1D'])
        expected = 0
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_1D_2(self):
        kernel = Epanechnikov(dimension=1)
        actual = kernel.evaluate(self.data['u_2_1D'])
        # TODO compute
        expected = None
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_2D_1(self):
        kernel = Epanechnikov(
            dimension=1,
            multivariate_approach=epanechnikov.Multivariate.multiplication)
        actual = kernel.evaluate(self.data['u_1_2D'])
        expected = 0
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_2D_2(self):
        kernel = Epanechnikov(
            dimension=1,
            multivariate_approach=epanechnikov.Multivariate.multiplication)
        actual = kernel.evaluate(self.data['u_2_2D'])
        # TODO compute
        expected = None
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_2D_3(self):
        kernel = Epanechnikov(
            dimension=1,
            multivariate_approach=epanechnikov.Multivariate.norm)
        actual = kernel.evaluate(self.data['u_1_2D'])
        # TODO compute
        expected = None
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_2D_4(self):
        kernel = Epanechnikov(
            dimension=1,
            multivariate_approach=epanechnikov.Multivariate.norm)
        actual = kernel.evaluate(self.data['u_2_2D'])
        # TODO compute
        expected = None
        self.assertAlmostEqual(actual, expected)