from __future__ import division

import kde._kde as _kde
import numpy as np
import scipy.interpolate as interpolate
import scipy.stats.mstats as stats

import kde.utils as utils
import kde.utils.automaticWindowWidthMethods as automaticWindowWidthMethods
from kde.estimatorimplementation import EstimatorImplementation
from kde.kernels.epanechnikov import Epanechnikov
from kde.parzen import ParzenEstimator, _ParzenEstimator_C


class MBEstimator(object):
    """Implementation of the Modifeid Breiman EstimatorImplementation, as proposed by Wilkinson and Meijer.
    """

    default_number_of_grid_points = 50

    def __init__(self, dimension, kernel_class=None, sensitivity=1 / 2, kernel_radius_fraction=1/4,
                 pilot_kernel_class=None,
                 pilot_window_width_method=automaticWindowWidthMethods.ferdosi,
                 number_of_grid_points=default_number_of_grid_points,
                 pilot_estimator_implementation=None, final_estimator_implementation=None):
        """ Init method of the Modified Breiman EstimatorImplementation.
        :param dimension: (int) The dimension of the data points of which the density is estimated.
        :param kernel_class: (kernel_class, optional) The kernel_class to use for the final density estimate, defaults to Gaussian.
        :param sensitivity: (int, optional) The sensitivity of the kernel_class method, defaults to 0.5.
        :param pilot_kernel_class: (kernel_class, optional) The kernel_class to use for the pilot density estimate, defaults to Epanechnikov_Python.
        :param pilot_window_width_method: (function, optional) The method to use for the estimation of the automatic
            window width. Defaults to ferdosi.
        :param number_of_grid_points: (int or list, optional) The number of grid points per dimension. If an int is
        passed the same number of grid points is used for each dimension.
        Defaults to *MBEstimator.default_number_of_grid_points*
        :param final_estimator_implementation: Class that inherits from EstimatorImplementation.
        """
        self._dimension = dimension
        self._general_window_width_method = pilot_window_width_method
        self._sensitivity = sensitivity
        self._pilot_kernel_class = pilot_kernel_class or Epanechnikov
        self._kernel = kernel_class() if kernel_class else Epanechnikov()
        self._number_of_grid_points = number_of_grid_points
        self._kernel_radius_fraction = kernel_radius_fraction

        self._pilot_estimator_implementation = pilot_estimator_implementation or _ParzenEstimator_C
        self._final_estimator_implementation = final_estimator_implementation or _MBEEstimator_C

    def estimate(self, xi_s, x_s=None):
        """
        Estimate the density of the points xi_s, use the points x_s to determine the density.
        :param xi_s: (array like) The data points to estimate the density for.
        :param x_s: (array like, optional) The data points to use to estimate the density. Defaults to xi_s.
        :return: The estimated densities of x_s.
        """
        if x_s is None:
            x_s = xi_s

        # Compute general window width
        general_window_width = self._general_window_width_method(xi_s)

        # Compute pilot densities
        pilot_densities = self._estimate_pilot_densities(general_window_width, xi_s=xi_s)

        # Compute local bandwidths
        local_bandwidths = self._compute_local_bandwidths(pilot_densities)

        # Compute densities
        estimator = self._final_estimator_implementation(xi_s=xi_s, x_s=x_s,
                                                         dimension=self._dimension,
                                                         kernel=self._kernel,
                                                         local_bandwidths=local_bandwidths,
                                                         general_bandwidth=general_window_width)
        densities = estimator.estimate()
        return densities

    def _compute_cell_size(self, bandwidth):
        return self._kernel_radius_fraction * self._pilot_kernel_class().radius(bandwidth)

    def _estimate_pilot_densities(self, general_bandwidth, xi_s):
        cell_size = self._compute_cell_size(general_bandwidth)

        # Compute grid for pilot densities
        grid_points = utils.Grid.cover(
            xi_s,
            number_of_grid_points=self._number_of_grid_points,
            cell_size=cell_size
        ).grid_points

        # Compute densities of the points on the grid
        pilot_estimator = ParzenEstimator(
            dimension=self._dimension,
            bandwidth=general_bandwidth, kernel_class=self._pilot_kernel_class,
            estimator_implementation=self._pilot_estimator_implementation)
        grid_densities = pilot_estimator.estimate(xi_s=xi_s, x_s=grid_points)

        # Interpolation: note it is not bilinear interpolation, but uses a triangulation
        pilot_densities = interpolate.griddata(grid_points, grid_densities, xi_s, method='linear')

        return pilot_densities

    def _compute_local_bandwidths(self, pilot_densities):
        geometric_mean = stats.gmean(pilot_densities)
        return np.power((pilot_densities / geometric_mean), - self._sensitivity)


class _MBEEstimator(EstimatorImplementation):

    def __init__(self, xi_s, x_s, dimension, kernel, local_bandwidths, general_bandwidth):
        super(_MBEEstimator, self).__init__(
            xi_s=xi_s, x_s=x_s, dimension=dimension,
            general_bandwidth=general_bandwidth, kernel=kernel)
        self._local_bandwidths = local_bandwidths.astype(float, copy=False)


class _MBEEstimator_Python(_MBEEstimator):
    def __init__(self, xi_s, x_s, dimension, kernel, local_bandwidths, general_bandwidth):
        super(_MBEEstimator_Python, self).__init__(
            xi_s=xi_s, x_s=x_s, dimension=dimension,
            general_bandwidth=general_bandwidth, local_bandwidths=local_bandwidths,
            kernel=kernel)

    def estimate(self):
        densities = np.empty(self.num_x_s)
        for idx, x in enumerate(self._x_s):
            densities[idx] = self._estimate_pattern(x)
        return (1 / self.num_xi_s) * densities

    def _estimate_pattern(self, x):
        factors = np.power(self._local_bandwidths * self._general_bandwidth, - self._dimension)
        terms = self._kernel.evaluate(
            np.divide(
                (x - self._xi_s).transpose(),
                self._general_bandwidth * self._local_bandwidths).transpose()
        )
        terms *= factors
        density = terms.sum()
        return density


class _MBEEstimator_C(_MBEEstimator):
    def __init__(self, xi_s, x_s, dimension, kernel, local_bandwidths, general_bandwidth):
        super(_MBEEstimator_C, self).__init__(
            xi_s=xi_s, x_s=x_s, dimension=dimension,
            general_bandwidth=general_bandwidth, local_bandwidths=local_bandwidths,
            kernel=kernel)

    def estimate(self):
        densities = np.empty(self.num_x_s, dtype=float)
        _kde.modified_breiman(self._x_s, self._xi_s,
                              self._general_bandwidth, self._local_bandwidths,
                              self._kernel.to_C_enum(),
                              densities)
        return densities