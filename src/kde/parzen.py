import math

import numpy as np

import kde.kernels as kernels


class Parzen:

    def __init__(self, window_width, dimension, kernel):
        self._kernel = kernel
        self._window_width = window_width
        self._dimension = dimension

    def estimate_python(self, xi_s, x_s=None):
        if x_s is None:
            x_s = xi_s
        (n, _) = xi_s.shape

        self._kernel.center = np.zeros(self._dimension)
        self._kernel.shape = np.identity(self._dimension)

        factor = 1 / (n * math.pow(self._window_width, self._dimension))
        (n_x, _) = x_s.shape
        densities = np.empty(n_x)
        for idx, x in enumerate(x_s):
            bump_sum = 0
            for xi in xi_s:
                bump_sum += self._kernel.evaluate((x - xi) / self._window_width)
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
    densities = estimator.estimate_python(xi_s=patterns)


class _EstimatorVectorized:

    def __init__(self, parent, xi_s, x_s):
        self._parent = parent
        self.xi_s = xi_s
        self.x_s = x_s
        (self.n_xi_s, _) = xi_s.shape
        (self.n_x_s, _) = x_s.shape

    def estimate(self):
        densities = np.empty(self.n_x_s)
        for idx, x in enumerate(self.n_x_s):
            densities[idx] = self._estimate_pattern(x)
        return densities

    def _estimate_pattern(self, x):

