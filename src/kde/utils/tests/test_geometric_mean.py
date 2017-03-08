from unittest import TestCase

import numpy as np
import scipy.stats.mstats as stats

import kde.utils._utils as _utils


class TestGeometricMean(TestCase):

    def test_compute_geometric_mean_1(self):
        data = np.array([2, 3, 4, 5.0])
        expected = stats.gmean(data)
        actual = _utils.geometric_mean(data)
        self.assertAlmostEqual(expected, actual)

    def test_compute_geometric_mean_2(self):
        data = np.array([2.5, 3.7, 4.8])
        expected = stats.gmean(data)
        actual = _utils.geometric_mean(data)
        self.assertAlmostEqual(expected, actual)