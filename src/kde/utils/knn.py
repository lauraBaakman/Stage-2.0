from sklearn.neighbors import KDTree
import numpy as np

import kde.utils.distanceMatrix as distanceMatrix
import kde.utils._utils as _utils


class KNN(object):

    def __init__(self, patterns, implementation=None):
        implementation_class = implementation or _KNN_C
        (self._num_patterns, _) = patterns.shape
        self._implementation = implementation_class(patterns=patterns)

    def find_k_nearest_neighbours(self, pattern, k):
        self._validate_k(k)
        return self._implementation.find_k_nearest_neighbours(pattern=pattern, k=k)

    def _validate_k(self, k):
        if k <= 0:
            raise TypeError("K should be greater than zero, not {}".format(k))
        if k > self._num_patterns:
            raise TypeError("K should be smaller than the number of patterns ({}), not {}".format(
                self._num_patterns, k)
            )


class _KNN(object):

    def __init__(self, patterns):
        self._patterns = patterns

    def find_k_nearest_neighbours(self, pattern, k):
        raise NotImplementedError()


class _KNN_C(_KNN):

    def __init__(self, patterns):
        super(_KNN_C, self).__init__(patterns)
        self._distance_matrix = distanceMatrix.compute_distance_matrix(patterns)

    def find_k_nearest_neighbours(self, pattern, k):
        (dimension,) = pattern.shape
        nearest_neighbours = np.empty([k, dimension], dtype=np.float64)
        pattern_idx = self._find_idx_of_pattern(pattern)
        _utils.knn(k, pattern_idx, self._patterns, self._distance_matrix, nearest_neighbours)
        return nearest_neighbours

    def _find_idx_of_pattern(self, pattern):
        idx_array = np.where(np.all(self._patterns == pattern, axis=1))
        return idx_array[0]


class _KNN_Python(_KNN):

    def __init__(self, patterns):
        super(_KNN_Python, self).__init__(patterns)
        self._kdtree = KDTree(patterns, metric='euclidean')

    def find_k_nearest_neighbours(self, pattern, k):
        indices = self._kdtree.query(pattern, k=k, return_distance=False)[0]
        return self._patterns[indices]
