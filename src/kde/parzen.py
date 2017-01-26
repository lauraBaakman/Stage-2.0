import math

import numpy as np

import kde._kde as kde
from kde.estimator import Estimator


class Parzen(Estimator):
    """
    Wrapper for the C implementation of Parzen density estimation with a Gaussian kernel.
    """

    def __init__(self, window_width, *args, **kwargs):
        """ Init method of the Parzen Estimator with a Gaussian kernel.
        :param window_width: (int) The window width to use.
        """
        super(Estimator, self).__init__()
        self._window_width = window_width

    def estimate(self, xi_s, x_s=None):
        """Estimate the density of the points xi_s, use the points x_s to determine the density.
        :param xi_s: (array like) The data points to estimate the density for.
        :param x_s: (array like, optional) The data points to use to estimate the density. Defaults to xi_s.
        :return: The estimated densities of x_s.
        """

        if x_s is None:
            x_s = xi_s
        self._validate_data(x_s, xi_s)
        (num_patterns, _) = xi_s.shape
        densities = np.empty(num_patterns, dtype=float)
        kde.parzen_standard_gaussian(x_s, xi_s, self._window_width, densities)
        return densities


class _ParzenInPython:
    """Implementation of the Parzen Estimator.
    """

    def __init__(self, dimension, window_width, kernel=None):
        """ Init method of the Parzen Estimator.
        :param dimension: (int) The dimension of the data points of which the density is estimated.
        :param window_width: (int) The window width to use.
        :param kernel: (kernel, optional) The kernel to use for the final density estimate, defaults to Gaussian.
        """
        self._dimension = dimension
        self._kernel = kernel or kernels.Gaussian()
        self._window_width = window_width

    def estimate(self, xi_s, x_s=None):
        """Estimate the density of the points xi_s, use the points x_s to determine the density.
        :param xi_s: (array like) The data points to estimate the density for.
        :param x_s: (array like, optional) The data points to use to estimate the density. Defaults to xi_s.
        :return: The estimated densities of x_s.
        """
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
    def num_xi_s(self):
        (n, _) = self._xi_s.shape
        return n

    @property
    def num_x_s(self):
        (n, _) = self._x_s.shape
        return n

    def estimate(self):
        self._kernel.center = np.zeros(self._dimension)
        self._kernel.shape = np.identity(self._dimension)

        densities = np.empty(self.num_x_s)
        factor = 1 / (self.num_xi_s * math.pow(self._window_width, self._dimension))
        for idx, x in enumerate(self._x_s):
            densities[idx] = self._estimate_pattern(x, factor)
        return densities

    def _estimate_pattern(self, x, factor):
        terms = self._kernel.evaluate((x - self._xi_s)/self._window_width)
        density = factor * terms.sum()
        return density
