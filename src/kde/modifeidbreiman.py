import scipy.stats.mstats as stats
import numpy as np

import kde
import kde.windowWidthMethods
import kde.kernels as kernels


class ModifiedBreimanEstimator(object):

    def __init__(self, dimension, kernel=None, sensitivity=1/2,
                 pilot_kernel=None,
                 pilot_window_width_method=kde.windowWidthMethods.ferdosi):
        self._dimension = dimension
        self._pilot_window_width_method = pilot_window_width_method
        self._sensitivity = sensitivity
        self._pilot_kernel = pilot_kernel or kernels.Epanechnikov(dimension=self._dimension)
        self._kernel = kernel or kernels.Gaussian()

    def estimate(self, xi_s, x_s=None):
        if x_s is None:
            x_s = xi_s

        # Compute pilot window width
        window_width = self._pilot_window_width_method(xi_s)

        # Compute grid for pilot densities
        grid_points = x_s

        # Compute pilot densities
        pilot_densities = kde.Parzen(
            window_width=window_width,
            dimension=self._dimension,
            kernel=self._pilot_kernel
        ).estimate(xi_s=xi_s, x_s=grid_points)

        # Multivariate Linear Interpolation
        #TODO do multivariate linear interpolation on pilot densitites

        # Compute local bandwidths
        local_bandwidths = self._compute_local_bandwidths(pilot_densities)

        # Compute densities
        estimator =_MBEstimator(xi_s=xi_s, x_s=x_s,
                                dimension=self._dimension,
                                kernel=self._kernel, local_bandwidths=local_bandwidths)
        densities = estimator.estimate()

    def _compute_local_bandwidths(self, pilot_densities):
        geometric_mean = stats.gmean(pilot_densities)
        return np.power((pilot_densities / geometric_mean), 1 / self._sensitivity)

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)


class _MBEstimator:

    def __init__(self, xi_s, x_s, dimension, kernel, local_bandwidths):
        self._xi_s = xi_s
        self._x_s = x_s
        self._dimension = dimension
        self._kernel = kernel
        self._local_bandwidths = local_bandwidths

    def estimate(self):
        raise NotImplementedError()
