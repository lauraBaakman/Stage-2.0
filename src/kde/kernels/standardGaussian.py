import numpy as np

import kde.kernels._kernels as _kernels


class StandardGaussian:
    """Implementation of the Standard Gaussian Kernel, i.e. a mean 0 and I as covariance matrix.
    """

    def __init__(self, *args, **kwargs):
        pass

    def evaluate(self, xs):
        if xs.ndim == 1:
            data = np.array([xs])
            density = _kernels.standard_gaussian_single_pattern(data)
            return density
        elif xs.ndim == 2:
            (num_patterns, _) = xs.shape
            densities = np.empty(num_patterns, dtype=float)
            _kernels.standard_gaussian_multi_pattern(xs, densities)
            return densities
        else:
            raise TypeError("Expected a vector or a matrix, not a {}-dimensional array.".format(xs.ndim))