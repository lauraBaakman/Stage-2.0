from unittest import TestCase

import numpy as np

from datasets.simulated.components import MultivariateGaussian, UnivariateGaussian


class TestMultivariateGaussian(TestCase):

    def setUp(self):
        super().setUp()
        mean3D = np.array([50, 50, 50])
        covariance3D = np.diag(np.array([30, 30, 30]))
        self._component3D = MultivariateGaussian(mean=mean3D,
                                                 covariance_matrix=covariance3D)

        mean2D = np.array([10, 20])
        covariance2D = np.diag(np.array([5, 10]))
        self._component2D = MultivariateGaussian(mean=mean2D,
                                                 covariance_matrix=covariance2D)

    def test_patterns_patterns_shape_0(self):
        expected_num_patterns = 1
        expected_dimension = 3

        actual_patterns = self._component3D.patterns(expected_num_patterns)
        (actual_num_patterns, actual_dimension) = actual_patterns.shape

        self.assertEqual(expected_num_patterns, actual_num_patterns)
        self.assertEqual(expected_dimension, actual_dimension)

    def test_patterns_patterns_shape_1(self):
        expected_num_patterns = 10
        expected_dimension = 3

        actual_patterns = self._component3D.patterns(expected_num_patterns)
        (actual_num_patterns, actual_dimension) = actual_patterns.shape

        self.assertEqual(expected_num_patterns, actual_num_patterns)
        self.assertEqual(expected_dimension, actual_dimension)

    def test_patterns_patterns_shape_2(self):
        expected_num_patterns = 10
        expected_dimension = 2

        actual_patterns = self._component2D.patterns(expected_num_patterns)
        (actual_num_patterns, actual_dimension) = actual_patterns.shape

        self.assertEqual(expected_num_patterns, actual_num_patterns)
        self.assertEqual(expected_dimension, actual_dimension)

    def test_densities_shape_0(self):
        expected_num_patterns = 1

        patterns = np.random.random((expected_num_patterns, 3))

        actual_densities = self._component3D.densities(patterns)
        (actual_num_patterns,) = actual_densities.shape
        self.assertEqual(actual_num_patterns, expected_num_patterns)

    def test_densities_shape_1(self):

        expected_num_patterns = 10

        patterns = np.random.random((expected_num_patterns, 3))

        actual_densities = self._component3D.densities(patterns)
        (actual_num_patterns, ) = actual_densities.shape
        self.assertEqual(actual_num_patterns, expected_num_patterns)

    def test_densities_shape_2(self):

        expected_num_patterns = 10

        patterns = np.random.random((expected_num_patterns, 2))

        actual_densities = self._component2D.densities(patterns)
        (actual_num_patterns, ) = actual_densities.shape
        self.assertEqual(actual_num_patterns, expected_num_patterns)

    def test_densities_values_0(self):
        patterns = np.array([[0.6324, 0.9649, 0.8003]])

        expected_densities = np.array([
            0.105040766084006e-55,
        ])
        actual_densities = self._component3D.densities(patterns)
        np.testing.assert_array_almost_equal(expected_densities, actual_densities)

    def test_densities_values_1(self):
        patterns = np.array([[0.8147, 0.0975, 0.1576],
                             [0.9058, 0.2785, 0.9706],
                             [0.1270, 0.5469, 0.9572],
                             [0.9134, 0.9575, 0.4854],
                             [0.6324, 0.9649, 0.8003]])

        expected_densities = np.array([
            0.011737828503672e-55,
            0.070254716194359e-55,
            0.029642020542178e-55,
            0.098033561414839e-55,
            0.105040766084006e-55,
        ])
        actual_densities = self._component3D.densities(patterns)
        np.testing.assert_array_almost_equal(expected_densities, actual_densities)

    def test_densities_values_2(self):
        patterns = np.array([
            [18.001538807704019, 19.800602983147598],
            [16.192649458227702, 22.260215517533140],
            [6.981561039870828, 19.351840412745855],
            [16.786295177227323, 19.607421100999392],
            [11.622053158145041, 24.710837465506252],
        ])

        expected_densities = np.array([
            0.000037232115220,
            0.000376644792486,
            0.008861923068881,
            0.000223304398417,
            0.005703965209363,
        ])
        actual_densities = self._component2D.densities(patterns)
        np.testing.assert_array_almost_equal(expected_densities, actual_densities)


class TestUnivariateGaussian(TestCase):

    def setUp(self):
        super().setUp()
        self._mean = 10
        self._variance = 0.5
        self._component = UnivariateGaussian(mean=self._mean, variance=self._variance)

    def test_patterns_shape_0(self):
        expected_num_patterns = 1
        expected_dimension = 1

        actual_patterns = self._component.patterns(expected_num_patterns)
        (actual_num_patterns, actual_dimension) = actual_patterns.shape

        self.assertEqual(expected_num_patterns, actual_num_patterns)
        self.assertEqual(expected_dimension, actual_dimension)

    def test_patterns_shape_1(self):
        expected_num_patterns = 10
        expected_dimension = 1

        actual_patterns = self._component.patterns(expected_num_patterns)
        (actual_num_patterns, actual_dimension) = actual_patterns.shape

        self.assertEqual(expected_num_patterns, actual_num_patterns)
        self.assertEqual(expected_dimension, actual_dimension)

    def test_densities_shape_0(self):
        expected_num_patterns = 1

        patterns = np.random.random((expected_num_patterns, 1))

        actual_densities = self._component.densities(patterns)
        (actual_num_patterns,) = actual_densities.shape

        self.assertEqual(actual_num_patterns, expected_num_patterns)

    def test_densities_shape_1(self):
        expected_num_densities = 8
        patterns = np.random.random((expected_num_densities, 1))

        actual_densities = self._component.densities(patterns)

        (actual_num_densities,) = actual_densities.shape

        self.assertEqual(expected_num_densities, actual_num_densities)

    def test_densities_values_0(self):
        pattern = np.array([
            7.5,
        ])
        expected_density = np.array([
            0.001089142115176
        ])

        actual_density = self._component.densities(pattern)

        np.testing.assert_array_almost_equal(expected_density, actual_density)

    def test_densities_values_1(self):
        pattern = np.array([
            7.5,
            10.5,
            9.5
        ])
        expected_density = np.array([
            0.001089142115176,
            0.439391289467722,
            0.439391289467722,
        ])

        actual_density = self._component.densities(pattern)

        np.testing.assert_array_almost_equal(expected_density, actual_density)