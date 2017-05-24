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

    _square_root_of_the_variance = np.sqrt(16.0 / 21.0)

    def __init__(self):
        super(_Epanechnikov_Python, self).__init__()

    def evaluate(self, x):
        x *= 1.0 / self._square_root_of_the_variance
        if x.ndim == 1:
            density = self._evaluate_single_pattern(x)
            return np.array([density])
        elif x.ndim == 2:
            return self._evaluate_multiple_patterns(x)
        else:
            raise TypeError("Expected a vector or a matrix, not a {}-dimensional array.".format(x.ndim))

    def _compute_unit_variance_factor(self, dimension):
        return np.power(1.0 / self._square_root_of_the_variance, dimension)

    def _evaluate_single_pattern(self, pattern):
        dimension = pattern.size
        unit_constant = self._compute_unit_variance_factor(dimension)
        volume = self._unit_sphere_volume(dimension)
        return self._evaluate_single_pattern_with_constants(pattern=pattern, dimension=dimension,
                                                            volume=volume, unit_constant=unit_constant)

    def _evaluate_multiple_patterns(self, patterns):
        def evaluate(pattern, dimension, volume):
            dot_product = np.dot(pattern, pattern)
            if dot_product >= 1:
                return 0
            return (dimension + 2) / (2 * volume) * (1 - dot_product)

        (num_patterns, dimension) = patterns.shape
        volume = self._unit_sphere_volume(dimension)
        densities = np.empty(num_patterns)
        unit_constant = self._compute_unit_variance_factor(dimension)
        for idx, pattern in enumerate(patterns):
            densities[idx] = self._evaluate_single_pattern_with_constants(pattern=pattern, dimension=dimension,
                                                                          volume=volume, unit_constant=unit_constant)
        return densities

    def _evaluate_single_pattern_with_constants(self, pattern, dimension, volume, unit_constant):
        dot_product = np.dot(pattern, pattern)
        if dot_product >= 1:
            return 0
        return unit_constant * ((dimension + 2) / (2 * volume)) * (1 - dot_product)

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
