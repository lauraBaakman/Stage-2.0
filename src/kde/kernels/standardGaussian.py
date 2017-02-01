import kde.kernels._kernels as _kernels
import numpy as np
import scipy.stats as stats

from kde.kernels.kernel import Kernel


class StandardGaussian(Kernel):
    def __init__(self, implementation=None):
        implementation_class = implementation or _StandardGaussian_C
        self._implementation = implementation_class()

    def evaluate(self, pattern):
        return self._implementation.evaluate(pattern)

    def to_C_enum(self):
        return 1


class _StandardGaussian_C(StandardGaussian):
    def __init__(self):
        pass

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


class _StandardGaussian_Python(StandardGaussian):
    def __init__(self):
        pass

    def evaluate(self, xs):
        dimension = self._get_data_dimension(xs)
        mean = np.zeros(dimension)
        covariance = np.identity(dimension)
        return stats.multivariate_normal(mean=mean, cov=covariance).pdf(xs)

    def _get_data_dimension(self, xs):
        if xs.ndim == 1:
            (dimension,) = xs.shape
        elif xs.ndim == 2:
            (_, dimension) = xs.shape
        else:
            raise TypeError("Expected a vector or a matrix, not a {}-dimensional array.".format(xs.ndim))
        return dimension
