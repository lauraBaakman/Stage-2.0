import math
from unittest import TestCase

import numpy as np

import kde.automaticWindowWidthMethods as sigma_methods


class TestSilverman(TestCase):

    def test__volume_nd_unit_sphere_1(self):
        expected = sigma_methods._volume_nd_unit_sphere(1)
        actual = 2
        self.assertAlmostEqual(expected, actual)

    def test__volume_nd_unit_sphere_2(self):
        expected = sigma_methods._volume_nd_unit_sphere(2)
        actual = math.pi
        self.assertAlmostEqual(expected, actual)

    def test__volume_nd_unit_sphere_3(self):
        expected = sigma_methods._volume_nd_unit_sphere(3)
        actual = 4 / 3 * math.pi
        self.assertAlmostEqual(expected, actual)

    def test__volume_nd_unit_sphere_4(self):
        expected = sigma_methods._volume_nd_unit_sphere(4)
        actual = (math.pi * math.pi) / 2
        self.assertAlmostEqual(expected, actual)

    def test__volume_nd_unit_sphere_5(self):
        expected = sigma_methods._volume_nd_unit_sphere(4)
        actual = (8 * math.pi * math.pi) / 15
        self.assertAlmostEqual(expected, actual)

    def test_silverman(self):
        datapoints = np.array([
            [2, 2],
            [3, 4],
            [4, 6]
        ])
        expected = 3.333333333333333
        actual = sigma_methods.silverman(datapoints)
        self.assertAlmostEqual(expected, actual)