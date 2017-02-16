from sklearn.neighbors import KDTree

import kde.utils.distanceMatrix as distanceMatrix


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
        raise NotImplementedError()


class _KNN_Python(_KNN):

    def __init__(self, patterns):
        super(_KNN_Python, self).__init__(patterns)
        self._kdtree = KDTree(patterns, metric='euclidean')

    def find_k_nearest_neighbours(self, pattern, k):
        indices = self._kdtree.query(pattern, k=k, return_distance=False)[0]
        return self._patterns[indices]

