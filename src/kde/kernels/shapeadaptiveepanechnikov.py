import math

import numpy as np
from numpy import linalg as LA
import scipy.special


import kde.kernels._kernels as _kernels
from kde.kernels.kernel import ShapeAdaptiveKernel_C, ShapeAdaptiveKernel_Python, ShapeAdaptive

_as_c_enum = 3


class ShapeAdaptiveEpanechnikov(ShapeAdaptive):

    def __new__(cls, bandwidth_matrix, implementation=None):
        implementation_class = implementation or _ShapeAdaptiveEpanechnikov_C
        cls._validate_bandwidth_matrix(bandwidth_matrix)
        return implementation_class(bandwidth_matrix=bandwidth_matrix)

    @staticmethod
    def to_C_enum():
        return _as_c_enum


class _ShapeAdaptiveEpanechnikov_C(ShapeAdaptiveKernel_C):
    def __init__(self, *args, **kwargs):
        super(_ShapeAdaptiveEpanechnikov_C, self).__init__(*args, **kwargs)

    def to_C_enum(self):
        return _as_c_enum

    def _handle_single_pattern(self, x, local_bandwidth):
        # data = np.array(x, ndmin=2)
        # density = _kernels.sa_gaussian_single_pattern(data, local_bandwidth, self._global_bandwidth_matrix)
        # return density
        raise NotImplementedError()

    def _handle_multiple_patterns(self, xs, local_bandwidths):
        # (num_patterns, _) = xs.shape
        # densities = np.empty(num_patterns, dtype=float)
        # _kernels.sa_gaussian_multi_pattern(xs, local_bandwidths, self._global_bandwidth_matrix, densities)
        # return densities
        raise NotImplementedError()


class _ShapeAdaptiveEpanechnikov_Python(ShapeAdaptiveKernel_Python):
    _kernel_variance = 16.0 / 21.0
    _square_root_of_the_variance = np.sqrt(_kernel_variance)

    def __init__(self, bandwidth_matrix, *args, **kwargs):
        self._global_bandwidth_matrix = self._square_root_of_the_variance * bandwidth_matrix
        self._unit_sphere_volume = self._compute_unit_sphere_volume(self.dimension)

    @property
    def dimension(self):
        (d, _) = self._global_bandwidth_matrix.shape
        return d

    def to_C_enum(self):
        return _as_c_enum

    def _evaluate_pattern(self, pattern, local_bandwidth):
        local_bandwidth_matrix = local_bandwidth * self._global_bandwidth_matrix
        local_inverse = LA.inv(local_bandwidth_matrix)
        local_scaling_factor = 1 / LA.det(local_bandwidth_matrix)
        return local_scaling_factor * self._epanechnikov(np.matmul(pattern, local_inverse))

    def _epanechnikov(self, pattern):
        dot_product = np.dot(pattern, pattern)
        if dot_product < 1.0:
            return (2 + self.dimension) / (2 * self._unit_sphere_volume) * (1 - dot_product)
        return 0

    def _compute_unit_sphere_volume(self, dimension):
        numerator = math.pow(math.pi, dimension / 2.0)
        denominator = scipy.special.gamma(dimension / 2.0 + 1)
        return numerator / denominator
