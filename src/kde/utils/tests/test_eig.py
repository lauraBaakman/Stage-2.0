from unittest import TestCase

import numpy as np

from kde.utils.eigenvaluesvectors import eigenValuesAndVectors, _eig_C, _eig_Python

class TestEig(TestCase):
    def test_eig_Python(self):
        self.eig_test_helper(lambda data: eigenValuesAndVectors(data, _eig_Python))

    def test_eig_C(self):
        self.eig_test_helper(lambda data: eigenValuesAndVectors(data, _eig_C))

    def eig_test_helper(self, the_function):
        data = np.array([
            [0.0688, 0.0275, -0.0047, -0.0156],
            [0.0275, 0.1025, -0.0146, 0.0211],
            [-0.0047, -0.0146, 0.0841, -0.0420],
            [-0.0156, 0.0211, -0.0420, 0.1017]
        ])
        expected_values = np.array([-0.6227, -0.4299, -0.1935, 0.6674])
        expected_vectors = np.array([
            [-0.6227, 0.5103, -0.5884, 0.0743],
            [0.3384, -0.4299, -0.6671, 0.5055],
            [-0.4445, -0.6866, -0.1935, -0.5418],
            [-0.5479, -0.2886, 0.4138, 0.6674],
        ])

        actual_values, actual_vectors = the_function(data)

        np.testing.assert_array_almost_equal(expected_values, actual_values)
        np.testing.assert_array_almost_equal(expected_vectors, actual_vectors)
