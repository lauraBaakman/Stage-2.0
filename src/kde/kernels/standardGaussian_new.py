import numpy as np

import kde.kernels._kernels as _kernels


class StandardGaussian_New:

    def __init__(self, *args, **kwargs):
        pass

    def evaluate(self, xs):
        if xs.ndim == 1:
            num_patterns = 1
        elif xs.ndim == 2:
            (num_patterns, _) = xs.shape
        else:
            raise TypeError("Expected a vector or a matrix, not a {}-dimensional array.".format(xs.ndim))

        densities = np.empty(num_patterns, dtype=float)
        _kernels.standard_gaussian(xs, densities)
        return np.squeeze(densities)

if __name__ == '__main__':
    gaussian = StandardGaussian_New()
    xs = np.array([[1, 2, 3], [4, 5, 6]])
    print(gaussian.evaluate(xs=xs))
