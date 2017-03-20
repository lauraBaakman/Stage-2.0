import math

import numpy as np
import scipy.special

import kde.kernels._kernels as _kernels
from kde.kernels.kernel import Kernel

_as_C_enum = 2


class Epanechnikov(object):
    def __new__(cls, implementation=None):
        implementation_class = implementation or _Epanechnikov_C
        return implementation_class()

    @staticmethod
    def to_C_enum():
        return _as_C_enum

class _Epanechnikov(Kernel):

    def __init__(self):
        pass

    @staticmethod
    def to_C_enum():
        return _as_C_enum


class _Epanechnikov_Python(_Epanechnikov):

    def __init__(self):
        super(_Epanechnikov_Python, self).__init__()

    def evaluate(self, x):
        if x.ndim == 1:
            density = self._evaluate_single_pattern(x)
            return np.array([density])
        elif x.ndim == 2:
            return self._evaluate_multiple_patterns(x)
        else:
            raise TypeError("Expected a vector or a matrix, not a {}-dimensional array.".format(x.ndim))

    def _evaluate_single_pattern(self, pattern):
        dimension = pattern.size
        x_dot_product = np.dot(pattern, pattern)
        if x_dot_product >= 1:
            return 0
        return (dimension + 2) / (2 * self._unit_sphere_volume(dimension)) * (1 - x_dot_product)

    def _evaluate_multiple_patterns(self, patterns):
        def evaluate(pattern, dimension, volume):
            dot_product = np.dot(pattern, pattern)
            if dot_product >= 1:
                return 0
            return (dimension + 2) / (2 * volume) * (1 - dot_product)

        (num_patterns, dimension) = patterns.shape
        volume = self._unit_sphere_volume(dimension)
        densities = np.empty(num_patterns)
        for idx, pattern in enumerate(patterns):
            densities[idx] = evaluate(pattern=pattern, dimension=dimension, volume=volume)
        return densities

    def _unit_sphere_volume(self, dimension):
        numerator = math.pow(math.pi, dimension / 2.0)
        denominator = scipy.special.gamma(dimension / 2.0 + 1)
        return numerator / denominator


class _Epanechnikov_C(_Epanechnikov):

    def __init__(self):
        super(_Epanechnikov_C, self).__init__()

    def evaluate(self, x):
        if x.ndim == 1:
            data = np.array([x])
            density = _kernels.epanechnikov_single_pattern(data)
            return np.array([density])
        elif x.ndim == 2:
            (num_patterns, _) = x.shape
            densities = np.empty(num_patterns, dtype=float)
            _kernels.epanechnikov_multi_pattern(x, densities)
            return densities
        else:
            raise TypeError("Expected a vector or a matrix, not a {}-dimensional array.".format(x.ndim))
