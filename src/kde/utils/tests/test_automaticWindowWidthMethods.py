import math
from unittest import TestCase

import numpy as np

import kde.utils.automaticWindowWidthMethods as bandwidth_methods


class TestSilverman(TestCase):
    def test__volume_nd_unit_sphere_1(self):
        expected = bandwidth_methods._volume_nd_unit_sphere(1)
        actual = 2
        self.assertAlmostEqual(expected, actual)

    def test__volume_nd_unit_sphere_2(self):
        expected = bandwidth_methods._volume_nd_unit_sphere(2)
        actual = math.pi
        self.assertAlmostEqual(expected, actual)

    def test__volume_nd_unit_sphere_3(self):
        expected = bandwidth_methods._volume_nd_unit_sphere(3)
        actual = 4.0 / 3.0 * math.pi
        self.assertAlmostEqual(expected, actual)

    def test__volume_nd_unit_sphere_4(self):
        expected = bandwidth_methods._volume_nd_unit_sphere(4)
        actual = (math.pi * math.pi) / 2
        self.assertAlmostEqual(expected, actual)

    def test__volume_nd_unit_sphere_5(self):
        expected = bandwidth_methods._volume_nd_unit_sphere(5)
        actual = (8 * math.pi * math.pi) / 15
        self.assertAlmostEqual(expected, actual)

    def test__volume_nd_unit_sphere_6(self):
        expected = bandwidth_methods._volume_nd_unit_sphere(6)
        actual = (math.pi * math.pi * math.pi) / 6
        self.assertAlmostEqual(expected, actual)

    def test_silverman(self):
        datapoints = np.array([
            [2, 2],
            [3, 4],
            [4, 6]
        ])
        expected = 3.333333333333333
        actual = bandwidth_methods.silverman(datapoints)
        self.assertAlmostEqual(expected, actual)


class TestFerdosi(TestCase):
    def test_ferdosi(self):
        data_points = np.array([
            [15, 20, 35, 40, 50],
            [15, 20, 35, 40, 50]
        ]).transpose()
        expected = 14.290703494871073
        actual = bandwidth_methods.ferdosi(data_points)
        self.assertAlmostEqual(expected, actual)
