import math

import numpy as np

import kde.kernels as kernels


class Parzen:

    def __init__(self, window_width, dimension, kernel):
        self._kernel = kernel
        self._window_width = window_width
        self._dimension = dimension

    def estimate_python(self, x_is, x_s=None):
        if x_s is None:
            x_s = x_is
        (n, _) = x_is.shape

        self._kernel.center = np.zeros(self._dimension)
        self._kernel.shape = np.identity(self._dimension)

        factor = 1 / (n * math.pow(self._window_width, self._dimension))
        (n_x_i, _) = x_s.shape
        densities = np.empty(n_x_i)
        for idx, x in enumerate(x_s):
            bump_sum = 0
            for x_i in x_is:
                bump_sum += self._kernel.evaluate((x - x_i) / self._window_width)
            densities[idx] = factor * bump_sum
        return densities

    def estimate_python_vectorized(self, data, patterns_to_estimate=None):
        if patterns_to_estimate is None:
            patterns_to_estimate = data
        return None


def benchmark_python(n=1000, dimension=3):
    patterns = np.random.randn(n, dimension)
    window_width = 1 / np.sqrt(n)
    kernel_shape = window_width * window_width * np.identity(dimension)
    kernel = kernels.Gaussian(covariance_matrix=kernel_shape)
    estimator = Parzen(window_width=window_width, dimension=dimension, kernel=kernel)
    densities = estimator.estimate_python(x_is=patterns)