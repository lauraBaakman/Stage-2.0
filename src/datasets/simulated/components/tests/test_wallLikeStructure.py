from unittest import TestCase

import numpy as np

import datasets.simulated.components as components


class TestWallLikeStructure(TestCase):

    def setUp(self):
        super().setUp()
        self._min_value = 10
        self._max_value = 20
        x_component = components.UniformRandomNoise(minimum_value=self._min_value, maximum_value=self._max_value, dimension=1)
        y_component = components.UniformRandomNoise(minimum_value=self._min_value, maximum_value=self._max_value, dimension=1)
        z_component = components.UnivariateGaussian(mean=15, variance=2.0)
        self._component3D = components.WallLikeStructure(
            one_dimensional_components=[x_component, y_component, z_component]
        )
        self._component2D = components.WallLikeStructure(
            one_dimensional_components=[x_component, z_component]
        )

    def test_patterns_shape_3D_0(self):
        expected_num_patterns = 1
        expected_dimension = 3

        actual_patterns = self._component3D.patterns(expected_num_patterns)
        (actual_num_patterns, actual_dimension) = actual_patterns.shape

        self.assertEqual(expected_num_patterns, actual_num_patterns)
        self.assertEqual(expected_dimension, actual_dimension)

    def test_patterns_shape_2D_0(self):
        expected_num_patterns = 1
        expected_dimension = 2

        actual_patterns = self._component2D.patterns(expected_num_patterns)
        (actual_num_patterns, actual_dimension) = actual_patterns.shape

        self.assertEqual(expected_num_patterns, actual_num_patterns)
        self.assertEqual(expected_dimension, actual_dimension)

    def test_patterns_shape_3D_1(self):
        expected_num_patterns = 7
        expected_dimension = 3

        actual_patterns = self._component3D.patterns(expected_num_patterns)
        (actual_num_patterns, actual_dimension) = actual_patterns.shape

        self.assertEqual(expected_num_patterns, actual_num_patterns)
        self.assertEqual(expected_dimension, actual_dimension)

    def test_patterns_shape_2D_1(self):
        expected_num_patterns = 9
        expected_dimension = 2

        actual_patterns = self._component2D.patterns(expected_num_patterns)
        (actual_num_patterns, actual_dimension) = actual_patterns.shape

        self.assertEqual(expected_num_patterns, actual_num_patterns)
        self.assertEqual(expected_dimension, actual_dimension)

    def test_patterns_values_3D(self):
        patterns = self._component3D.patterns(10)
        self._in_range(patterns.transpose()[0])
        self._in_range(patterns.transpose()[1])

    def test_patterns_values_2D(self):
        patterns = self._component2D.patterns(10)
        self._in_range(patterns.transpose()[0])

    def test_densities_shape_3D_0(self):
        expected_num_patterns = 1

        patterns = np.random.random((expected_num_patterns, 3))

        actual_densities = self._component3D.densities(patterns)
        (actual_num_patterns,) = actual_densities.shape
        self.assertEqual(actual_num_patterns, expected_num_patterns)

    def test_densities_shape_2D_0(self):
        expected_num_patterns = 1

        patterns = np.random.random((expected_num_patterns, 2))

        actual_densities = self._component2D.densities(patterns)
        (actual_num_patterns,) = actual_densities.shape
        self.assertEqual(actual_num_patterns, expected_num_patterns)

    def test_densities_shape_3D_1(self):
        expected_num_patterns = 6

        patterns = np.random.random((expected_num_patterns, 3))

        actual_densities = self._component3D.densities(patterns)
        (actual_num_patterns,) = actual_densities.shape
        self.assertEqual(actual_num_patterns, expected_num_patterns)

    def test_densities_shape_2D_1(self):
        expected_num_patterns = 6

        patterns = np.random.random((expected_num_patterns, 2))

        actual_densities = self._component2D.densities(patterns)
        (actual_num_patterns,) = actual_densities.shape
        self.assertEqual(actual_num_patterns, expected_num_patterns)

    def test_densities_values_3D_0(self):
        pattern = np.array([
            [15, 15, 16]
        ])
        expected_density = np.array([
            0.002196956447339
        ])
        actual_density = self._component3D.densities(pattern)
        np.testing.assert_array_almost_equal(expected_density, actual_density)

    def test_densities_values_2D_0(self):
        pattern = np.array([
            [15, 16]
        ])
        expected_density = np.array([
            0.021969564473386
        ])
        actual_density = self._component2D.densities(pattern)
        np.testing.assert_array_almost_equal(expected_density, actual_density)

    def test_densities_values_3D_1(self):
        pattern = np.array([
            [8, 15, 16],
            [15, 8, 16],
            [15, 15, 16],
            [10, 15, 20],
            [15, 20, 10],
            [30, 15, 16],
            [15, 30, 16],
        ])
        expected_density = np.array([
            0.0,
            0.0,
            0.002196956447339,
            5.445710575881780e-06,
            5.445710575881780e-06,
            0.0,
            0.0,
        ])
        actual_density = self._component3D.densities(pattern)
        np.testing.assert_array_almost_equal(expected_density, actual_density)

    def test_densities_values_2D_1(self):
        pattern = np.array([
            [8, 16],
            [15, 16],
            [10, 20],
            [20, 20],
            [30, 16],
        ])
        expected_density = np.array([
            0.0,
            0.021969564473386,
            5.445710575881780e-05,
            5.445710575881780e-05,
            0.0,
        ])
        actual_density = self._component2D.densities(pattern)
        np.testing.assert_array_almost_equal(expected_density, actual_density)

    # noinspection PyTypeChecker
    def _in_range(self, values):
        self.assertTrue(np.all(values >= self._min_value))
        self.assertTrue(np.all(values <= self._max_value))