import math

import numpy as np

import kde.kernels as kernels


class Parzen:

    def __init__(self, window_width, dimension, kernel):
        self._kernel = kernel
        self._window_width = window_width
        self._dimension = dimension

    def estimate_python(self, data, patterns_to_estimate=None):
        self._kernel.center = np.zeros((self._dimension))
        self._kernel.shape = np.identity(self._dimension)
        if patterns_to_estimate is None:
            patterns_to_estimate = data
        (n, _) = data.shape
        factor = 1 / (n * (math.pow(self._window_width, self._dimension)))
        for x in patterns_to_estimate:
            sum = 0
            for x_i in data:
                sum += self._kernel.evaluate(1/self._window_width * (x - x_i))

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
    densities = estimator.estimate_python(data=patterns)