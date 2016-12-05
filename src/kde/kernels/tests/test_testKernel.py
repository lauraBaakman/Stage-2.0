from unittest import TestCase

import numpy as np

from kde.kernels.testKernel import TestKernel


class TestTestKernel(TestCase):

    def test_evaluate(self):
        x = np.array([[2, 3], [3, 4], [4, 5]])
        actual = TestKernel().evaluate(x)
        expected = np.array([2.5, 3.5, 4.5])
        np.testing.assert_array_equal(actual, expected)
