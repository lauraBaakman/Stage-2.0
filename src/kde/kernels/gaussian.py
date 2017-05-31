import numpy as np
import scipy.stats as stats

import kde.kernels._kernels as _kernels
from kde.kernels.kernel import Kernel, KernelException

_as_C_enum = 1


class Gaussian(Kernel):

    def __new__(cls, implementation=None):
        implementation_class = implementation or _StandardGaussian_C
        return implementation_class()

    @staticmethod
    def to_C_enum():
        return _as_C_enum


class _StandardGaussian(Kernel):

    def __init__(self):
        pass

    @staticmethod
    def to_C_enum():
        return _as_C_enum


class _StandardGaussian_C(_StandardGaussian):
    def __init__(self):
        super(_StandardGaussian_C, self).__init__()

    def evaluate(self, xs):
        if xs.ndim == 1:
            return self._handle_single_pattern(xs)
        elif xs.ndim == 2:
            return self._handle_multiple_patterns(xs)
        else:
            raise TypeError("Expected a vector or a matrix, not a {}-dimensional array.".format(xs.ndim))

    def _handle_single_pattern(self, xs):
        data = np.array([xs])
        density = _kernels.standard_gaussian_single_pattern(data)
        return density

    def _handle_multiple_patterns(self, xs):
        (num_patterns, _) = xs.shape
        densities = np.empty(num_patterns, dtype=float)
        _kernels.standard_gaussian_multi_pattern(xs, densities)
        return densities


class _StandardGaussian_Python(_StandardGaussian):
    def __init__(self):
        super(_StandardGaussian_Python, self).__init__()

    def evaluate(self, xs):
        dimension = self._get_data_dimension(xs)
        mean = np.zeros(dimension)
        covariance = np.identity(dimension)
        return stats.multivariate_normal(mean=mean, cov=covariance).pdf(xs)
