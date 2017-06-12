import warnings

from sklearn.neighbors import KDTree
import numpy as np


class KNN(object):

    def __init__(self, patterns):
        implementation_class = _KNN_Python
        (self._num_patterns, _) = patterns.shape
        self._patterns = patterns
        self._implementation = implementation_class(patterns=patterns)

    def find_k_nearest_neighbours(self, pattern, k):
        self._validate_k(k)
        try:
            neighbours = self._implementation.find_k_nearest_neighbours(pattern=pattern, k=k)
        except KNNException:
            warnings.warn("Switching to the Python implementation of KNN, the C implementation does not support KNN "
                          "with patterns that are not present in the distance matrix.")
            self._implementation = _KNN_Python(patterns=self._patterns)
            neighbours = self._implementation.find_k_nearest_neighbours(pattern=pattern, k=k)
        return neighbours

    def _validate_k(self, k):
        if k <= 0:
            raise KNNException("K should be greater than zero, not {}".format(k))
        if k > self._num_patterns:
            raise KNNException("K should be smaller than the number of patterns ({}), not {}".format(
                self._num_patterns, k)
            )

class _KNN(object):

    def __init__(self, patterns):
        self._patterns = patterns

    def find_k_nearest_neighbours(self, pattern, k):
        raise NotImplementedError()


class _KNN_Python(_KNN):

    def __init__(self, patterns):
        super(_KNN_Python, self).__init__(patterns)
        self._kdtree = KDTree(patterns, metric='euclidean')

    def find_k_nearest_neighbours(self, pattern, k):
        indices = self._kdtree.query(pattern.reshape(1, -1), k=k, return_distance=False)[0]
        return self._patterns[indices]


class KNNException(Exception):

    def __init__(self, message, number=None, *args):
        self.message = message
        self.number = number
        super(KNNException, self).__init__(message, *args)