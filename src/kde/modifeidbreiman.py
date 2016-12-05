import math

import scipy.stats.mstats as stats
import numpy as np

import kde
import kde.automaticWindowWidthMethods
import kde.kernels as kernels


class ModifiedBreimanEstimator(object):
    """Implementation of the Modifeid Breiman Estimator, as proposed by Wilkinson and Meijer.
    """

    def __init__(self, dimension, kernel=None, sensitivity=1/2,
                 pilot_kernel=None,
                 pilot_window_width_method=kde.automaticWindowWidthMethods.ferdosi):
        """ Init method of the Modified Breiman Estimator.
        :param dimension: (int) The dimension of the data points of which the density is estimated.
        :param kernel: (kernel, optional) The kernel to use for the final density estimate, defaults to Gaussian.
        :param sensitivity: (int, optional) The sensitivity of the kernel method, defaults to 0.5.
        :param pilot_kernel: (kernel, optional) The kernel to use for the pilot density estimate, defaults to Epanechnikov.
        :param pilot_window_width_method: (function, optional) The method to use for the estimation of the automatic
            window width.
        """
        self._dimension = dimension
        self._pilot_window_width_method = pilot_window_width_method
        self._sensitivity = sensitivity
        self._pilot_kernel = pilot_kernel or kernels.Epanechnikov(dimension=self._dimension)
        self._kernel = kernel or kernels.Gaussian()

    def estimate(self, xi_s, x_s=None):
        """
        Estimate the density of the points xi_s, use the points x_s to determine the density.
        :param xi_s: (array like) The data points to estimate the density for.
        :param x_s: (array like, optional) The data points to use to estimate the density. Defaults to xi_s.
        :return: The estimated densities of x_s.
        """
        if x_s is None:
            x_s = xi_s

        # Compute pilot window width
        pilot_window_width = self._pilot_window_width_method(xi_s)

        # Compute grid for pilot densities
        grid_points = x_s

        # Compute pilot densities
        pilot_densities = kde.Parzen(
            window_width=pilot_window_width,
            dimension=self._dimension,
            kernel=self._pilot_kernel
        ).estimate(xi_s=xi_s, x_s=grid_points)

        # Multivariate Linear Interpolation
        #TODO do multivariate linear interpolation on pilot densitites

        # Compute local bandwidths
        local_bandwidths = self._compute_local_bandwidths(pilot_densities)

        # Compute densities
        estimator = _MBEstimator(xi_s=xi_s, x_s=x_s,
                                 dimension=self._dimension,
                                 kernel=self._kernel, local_bandwidths=local_bandwidths,
                                 general_bandwidth=pilot_window_width)
        densities = estimator.estimate()

    def _compute_local_bandwidths(self, pilot_densities):
        geometric_mean = stats.gmean(pilot_densities)
        return np.power((pilot_densities / geometric_mean), 1 / self._sensitivity)

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)


class _MBEstimator:

    def __init__(self, xi_s, x_s, dimension, kernel, local_bandwidths, general_bandwidth):
        self._xi_s = xi_s
        self._x_s = x_s
        self._dimension = dimension
        self._kernel = kernel
        self._local_bandwidths = local_bandwidths
        self._general_bandwidth = general_bandwidth

    @property
    def num_x_s(self):
        (n, _) = self._x_s.shape
        return n

    @property
    def num_xi_s(self):
        (n, _) = self._xi_s.shape
        return n

    def estimate(self):
        densities = np.empty(self.num_x_s)
        for idx, x in enumerate(self._x_s):
            densities[idx] = self._estimate_pattern(x)
        return (1 / self.num_xi_s) * densities

    def _estimate_pattern(self, x):
        factors = np.power(self._local_bandwidths * self._general_bandwidth, - self._dimension)
        terms = self._kernel.evaluate(
            np.divide((x - self._xi_s).transpose(), self._general_bandwidth * self._local_bandwidths).transpose()
        )
        terms *= factors
        density = terms.sum()
        return density
