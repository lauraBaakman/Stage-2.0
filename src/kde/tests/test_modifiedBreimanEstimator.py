from unittest import TestCase

import numpy as np

import kde


class TestModifiedBreimanEstimator(TestCase):

    def setUp(self):
        super().setUp()
        self.estimator = kde.ModifiedBreimanEstimator(dimension=2, sensitivity=0.5)


    def test__compute_local_bandwidths(self):
        densities = np.array([1, 2, 3, 4, 5, 6])
        expected = np.array([0.334024188266401, 0.668048376532802, 1.002072564799204,
                            1.336096753065605, 1.670120941332006, 2.004145129598407])
        actual = self.estimator._compute_local_bandwidths(densities)

