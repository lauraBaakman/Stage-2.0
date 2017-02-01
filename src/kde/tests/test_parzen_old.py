from unittest import TestCase

import numpy as np

from kde.parzen_old import Parzen


class TestParzen(TestCase):

    def setUp(self):
        super().setUp()
        self._patterns = np.array([
                [2, 3],
                [1, 3],
                [-2, 2]
            ])
        self._parzen_estimated_densities = np.array([
                0.591741568117374,
                0.591741568117374,
                0.589462752192205
            ])
        self._window_width = 0.3

    def test_estimate(self):
        estimator = Parzen(window_width=self._window_width)
        actual = estimator.estimate(xi_s=self._patterns)
        expected = self._parzen_estimated_densities
        np.testing.assert_array_almost_equal(actual, expected)