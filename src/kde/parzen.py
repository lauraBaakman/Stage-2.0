import math

import numpy as np

import kde.kernels as kernels

class Parzen:

    def __init__(self, dimension, window_width, kernel=None):
        self._dimension = dimension
        self._kernel = kernel or kernels.Gaussian(dimension=self._dimension)
        self._window_width = window_width


    def estimate(self, xi_s, x_s=None):
        if x_s is None:
            x_s = xi_s
        estimator = _ParzenEstimator(
            xi_s=xi_s, x_s=x_s,
            dimension=self._dimension, kernel=self._kernel, window_width=self._window_width)
        return estimator.estimate()

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)


class _ParzenEstimator:

    def __init__(self, xi_s, x_s, dimension, kernel, window_width):
        self._xi_s = xi_s
        self._x_s = x_s
        self._dimension = dimension
        self._kernel = kernel
        self._window_width = window_width

    @property
    def n_xi_s(self):
        (n, _) = self._xi_s.shape
        return n

    @property
    def n_x_s(self):
        (n, _) = self._x_s.shape
        return n

    def estimate(self):
        self._kernel.center = np.zeros(self._dimension)
        self._kernel.shape = np.identity(self._dimension)

        densities = np.empty(self.n_x_s)
        factor = 1 / (self.n_xi_s * math.pow(self._window_width, self._dimension))
        for idx, x in enumerate(self._x_s):
            densities[idx] = self._estimate_pattern(x, factor)
        return densities

    def _estimate_pattern(self, x, factor):
        terms = self._kernel.evaluate((x - self._xi_s)/self._window_width)
        density = factor * terms.sum()
        return density
