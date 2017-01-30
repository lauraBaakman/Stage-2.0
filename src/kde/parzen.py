import math
import warnings

import kde._kde as _kde
import numpy as np

from kde import kernels
from kde.estimator import Estimator


class Parzen(object):
    """
    Wrapper for the C implementation of Parzen density estimation with a Gaussian kernel.
    """

    def __init__(self, window_width, *args, **kwargs):
        """ Init method of the Parzen Estimator with a Gaussian kernel.
        :param window_width: (int) The window width to use.
        """
        self._window_width = window_width

    def estimate(self, xi_s, x_s=None):
        """Estimate the density of the points xi_s, use the points x_s to determine the density.
        :param x_s: (array like) The data points to estimate the density for.
        :param xi_s: (array like, optional) The data points to use to estimate the density. Defaults to x_s.
        :return: The estimated densities of x_s.
        """

        if x_s is None:
            x_s = xi_s

        # Compute densities
        estimator = _ParzenEstimator(xi_s=xi_s, x_s=x_s, general_bandwidth=self._window_width)
        return estimator.estimate()


class _ParzenEstimator(Estimator):
    def __init__(self, xi_s, x_s, general_bandwidth):
        warnings.warn("No matter the passed arguments the Standard Gaussian Kernel is used.")
        (_, dimension) = xi_s.shape
        super(_ParzenEstimator, self).__init__(x_s=x_s, xi_s=xi_s,
                                               dimension=dimension,
                                               kernel=None, general_bandwidth=general_bandwidth)

    def estimate(self):
        (num_patterns, _) = self._x_s.shape
        densities = np.empty(num_patterns, dtype=float)
        _kde.parzen_standard_gaussian(self._x_s, self._xi_s, self._general_bandwidth, densities)
        return densities


class Parzen_Python(object):
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
        estimator = _ParzenEstimator_Python(
            xi_s=xi_s, x_s=x_s,
            dimension=self._dimension, kernel=self._kernel, window_width=self._window_width)
        return estimator.estimate()

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)


class _ParzenEstimator_Python(Estimator):

    def __init__(self, xi_s, x_s, dimension, kernel, window_width):
        super(_ParzenEstimator_Python, self).__init__(
            xi_s=xi_s, x_s=x_s,
            dimension=dimension,
            kernel=kernel, general_bandwidth=window_width
        )

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
        factor = 1 / (self.num_xi_s * math.pow(self._general_bandwidth, self._dimension))
        for idx, x in enumerate(self._x_s):
            densities[idx] = self._estimate_pattern(x, factor)
        return densities

    def _estimate_pattern(self, x, factor):
        terms = self._kernel.evaluate((x - self._xi_s) / self._general_bandwidth)
        density = factor * terms.sum()
        return density
