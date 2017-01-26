from unittest import TestCase

import numpy as np

from kde.kernels import Epanechnikov
import kde.kernels.epanechnikov as epanechnikov


class TestEpanechnikov(TestCase):

    def setUp(self):
        super().setUp()

    def test_evaluate_1D_1(self):
        kernel = Epanechnikov()
        x = np.array([5])
        actual = kernel.evaluate(x)
        expected = np.array([0])
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_1D_2(self):
        kernel = Epanechnikov()
        x = np.array([-0.5])
        actual = kernel.evaluate(x)
        expected = np.array([0.562500000000000])
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_1D_3(self):
        kernel = Epanechnikov()
        x = np.array([0.5])
        actual = kernel.evaluate(x)
        expected = np.array([0.562500000000000])
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_1D_4(self):
        kernel = Epanechnikov()
        x = np.array([[5], [0.5]])
        actual = kernel.evaluate(x)
        expected = np.array([0, 0.562500000000000])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_evaluate_2D_1(self):
        kernel = Epanechnikov()
        x = np.array([5, 0.5])
        actual = kernel.evaluate(x)
        expected = np.array([0])
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_2D_2(self):
        kernel = Epanechnikov()
        x = np.array([0.5, 0.5])
        actual = kernel.evaluate(x)
        expected = np.array([0.318309886183791])
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_2D_3(self):
        kernel = Epanechnikov()
        x = np.array([[0.3, 0.5], [0.5, 0.5]])
        actual = kernel.evaluate(x)
        expected = np.array([0.420169049762604, 0.318309886183791])
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_3D(self):
        kernel = Epanechnikov()
        x = np.array([0.5, 0.5, 0.5])
        actual = kernel.evaluate(x)
        expected = np.array([0.149207759148652  ])
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_4D(self):
        kernel = Epanechnikov()
        x = np.array([[0.3, 0.5, 0.5, 0.5]])
        actual = kernel.evaluate(x)
        expected = np.array([0.097268336296644])
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_5D(self):
        kernel = Epanechnikov()
        x = np.array([[0.3, 0.5, 0.5, 0.5, 0.2]])
        actual = kernel.evaluate(x)
        expected = np.array([0.079790432118341])
        self.assertAlmostEqual(actual, expected)
