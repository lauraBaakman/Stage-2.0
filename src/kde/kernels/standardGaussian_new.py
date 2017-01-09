import numpy as np

import kde.kernels._kernels as _kernels


class StandardGaussian_New:

    def __init__(self, *args, **kwargs):
        pass

    def evaluate(self, xs):
        if xs.ndim == 1:
            density = _kernels.standard_gaussian_single_pattern(xs)
            return density
        elif xs.ndim == 2:
            (num_patterns, _) = xs.shape
            densities = np.empty(num_patterns, dtype=float)
            _kernels.standard_gaussian_multi_pattern(xs, densities)
            return densities
        else:
            raise TypeError("Expected a vector or a matrix, not a {}-dimensional array.".format(xs.ndim))

if __name__ == '__main__':
    gaussian = StandardGaussian_New()
    xs = np.matrix(
        [
            np.array([0.5, 0.5, 0.5]),
            np.array([-0.75, -0.5, 0.1])
        ])
    print(xs.shape)
    print(gaussian.evaluate(xs=xs))
