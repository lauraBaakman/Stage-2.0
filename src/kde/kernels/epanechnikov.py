import math

import numpy as np
import scipy.special

import kde.kernels._kernels as _kernels
from kde.kernels.kernel import SymmetricKernel

_as_C_enum = 2


class Epanechnikov(SymmetricKernel):
    def __new__(cls, implementation=None):
        implementation_class = implementation or _Epanechnikov_C
        return implementation_class()

    @staticmethod
    def to_C_enum():
        return _as_C_enum


class _Epanechnikov(SymmetricKernel):

    def __init__(self):
        pass

    @staticmethod
    def radius(bandwidth):
        return np.sqrt(5) * bandwidth

    @staticmethod
    def to_C_enum():
        return _as_C_enum


class _Epanechnikov_Python(_Epanechnikov):

    _sqrt_five = 2.236067977499790

    def __init__(self):
        super(_Epanechnikov_Python, self).__init__()

    def evaluate(self, x):
        if x.ndim == 1:
            density = self._evaluate_single_pattern(x)
            return np.array([density])
        elif x.ndim == 2:
            return self._evaluate_multiple_patterns(x)
        else:
            raise TypeError(
                "Expected a vector or a matrix, not a {}-dimensional array."
                .format(x.ndim)
            )

    def _evaluate_single_pattern(self, pattern):
        dimension = pattern.size
        unit_constant = self._compute_unit_constant(dimension)
        volume_constant = self._compute_volume_constant(dimension)
        return self._evaluate_single_pattern_with_constants(
            pattern=pattern,
            volume_constant=volume_constant,
            unit_constant=unit_constant
        )

    def _evaluate_multiple_patterns(self, patterns):
        (num_patterns, dimension) = patterns.shape
        volume_constant = self._compute_volume_constant(dimension)
        densities = np.empty(num_patterns)
        unit_constant = self._compute_unit_constant(dimension)
        for idx, pattern in enumerate(patterns):
            densities[idx] = self._evaluate_single_pattern_with_constants(
                pattern=pattern,
                volume_constant=volume_constant, unit_constant=unit_constant)
        return densities

    def _compute_unit_constant(self, dimension):
        return np.power(self._sqrt_five,  - dimension)

    def _compute_volume_constant(self, dimension):
        unit_sphere_volume = self._unit_sphere_volume(dimension)
        return (2.0 + dimension) / (2.0 * unit_sphere_volume)

    def _unit_sphere_volume(self, dimension):
        numerator = math.pow(math.pi, dimension / 2.0)
        denominator = scipy.special.gamma(dimension / 2.0 + 1)
        return numerator / denominator

    def _evaluate_single_pattern_with_constants(self, pattern, volume_constant, unit_constant):
        dot_product = np.dot(pattern, pattern)
        if dot_product >= self._sqrt_five:
            return 0
        return unit_constant * volume_constant * (1 - (1 / 5.0 * dot_product))


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
            raise TypeError(
                "Expected a vector or a matrix, not a {}-dimensional array."
                .format(x.ndim)
            )
