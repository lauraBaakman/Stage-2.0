import numpy as np
import scipy.spatial as spatial

import kde.utils._utils as _utils

def compute_distance_matrix(patterns, implementation=None):
    actual_implementation = implementation or _compute_distance_matrix_C
    return actual_implementation(patterns)


def _compute_distance_matrix_Python(patterns):
    distance_euclidean = spatial.distance_matrix(patterns, patterns, p=2)
    distance_squared_euclidean = distance_euclidean ** 2
    return distance_squared_euclidean


def _compute_distance_matrix_C(patterns):
    (num_patterns, _) = patterns.shape
    distances = np.empty([num_patterns, num_patterns], dtype=np.float64)
    return _utils.distance_matrix(patterns, distances)
