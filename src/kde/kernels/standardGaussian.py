import numpy as np
import scipy.stats as stats

from kde.kernels.kernel import Kernel, KernelException
import kde.kernels._kernels as _kernels


class StandardGaussian(Kernel):

    def __new__(cls, implementation=None):
        implementation_class = implementation or _StandardGaussian_C
        return implementation_class()


class _StandardGaussian(Kernel):

    def __init__(self):
        pass

    @staticmethod
    def _validate_scaling_factors_parameters(general_bandwidth, eigen_values):
        if eigen_values is None:
            return
        if np.all(eigen_values == general_bandwidth):
            return
        raise KernelException("The StandardGaussian can only have a covariance matrix of the form:"
                              "general_bandwidth * I. Thus the eigen values should all be equal to the general "
                              " bandwidth.")

    def to_C_enum(self):
        return 1


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

    def scaling_factor(self, general_bandwidth, eigen_values=None):
        self._validate_scaling_factors_parameters(general_bandwidth=general_bandwidth, eigen_values=eigen_values)
        raise NotImplementedError("This class does not have an implementation of the scaling factor computation method.")


class _StandardGaussian_Python(_StandardGaussian):
    def __init__(self):
        super(_StandardGaussian_Python, self).__init__()

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

    def scaling_factor(self, general_bandwidth, eigen_values=None):
        self._validate_scaling_factors_parameters(general_bandwidth=general_bandwidth, eigen_values=eigen_values)
        return general_bandwidth * np.sqrt(general_bandwidth)