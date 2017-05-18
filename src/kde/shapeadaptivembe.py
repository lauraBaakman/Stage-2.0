import numpy as np
from numpy.ma.extras import cov

from kde.modifeidbreiman import ModifiedBreimanEstimator
import kde.utils.automaticWindowWidthMethods as automaticWindowWidthMethods
from kde.estimatorimplementation import EstimatorImplementation
from kde.parzen import ParzenEstimator
from kde.utils.knn import KNN
import kde.utils.covariance as covariance
from kde.kernels.shapeadaptivegaussian import ShapeAdaptiveGaussian
from kde.kernels.epanechnikov import Epanechnikov
import kde.kernels.scaling


class ShapeAdaptiveMBE(ModifiedBreimanEstimator):
    """
    Implementation of the shape adaptive modified Breiman Estimator.
    """

    def __init__(self, dimension, sensitivity=1 / 2,
                 pilot_window_width_method=automaticWindowWidthMethods.ferdosi,
                 number_of_grid_points=ModifiedBreimanEstimator.default_number_of_grid_points,
                 pilot_kernel_class=None, pilot_estimator_implementation=None,
                 kernel_class=None, final_estimator_implementation=None):
        self._dimension = dimension
        self._general_window_width_method = pilot_window_width_method
        self._sensitivity = sensitivity
        self._pilot_kernel_class = pilot_kernel_class or Epanechnikov
        self._kernel = kernel_class or ShapeAdaptiveGaussian
        self._number_of_grid_points = number_of_grid_points or ModifiedBreimanEstimator.default_number_of_grid_points

        self._pilot_estimator_implementation = pilot_estimator_implementation or ParzenEstimator
        self._final_estimator_implementation = final_estimator_implementation or _ShapeAdaptiveMBE_C


class _ShapeAdaptiveMBE(EstimatorImplementation):
    def __init__(self, xi_s, x_s, dimension, kernel, local_bandwidths, general_bandwidth):
        super().__init__(xi_s, x_s, dimension, kernel, general_bandwidth)
        self._local_bandwidths = local_bandwidths.astype(float, copy=False)
        self._knn = KNN(patterns=self._xi_s)
        self._k = self._compute_k(self.num_xi_s, self.dimension)
        self._kernel = None
        self._kernel_class = kernel or ShapeAdaptiveGaussian

    def _compute_k(self, num_patterns, dimension):
        potential_k = round(np.sqrt(num_patterns))
        return int(max(potential_k, dimension)) + 1

    def estimate(self):
        raise NotImplementedError()


class _ShapeAdaptiveMBE_C(_ShapeAdaptiveMBE):

    def estimate(self):
        raise NotImplementedError("The C implementation of the Shape Adaptive Modified Breiman is estimator has not "
                              "yet been written.")


class _ShapeAdaptiveMBE_Python(_ShapeAdaptiveMBE):

    def estimate(self):
        densities = np.empty(self.num_x_s)
        for idx, x in enumerate(self._x_s):
            densities[idx] = self._estimate_pattern(x)
        return densities

    def _estimate_pattern(self, pattern, kernel_shape=None):
        kernel_shape = kernel_shape if kernel_shape is not None else self._determine_kernel_shape(pattern)

        # Define the kernel
        kernel = self._kernel_class(kernel_shape)

        #Density estimation
        density = sum(map(kernel.evaluate, pattern - self._xi_s, self._local_bandwidths))

        density *= (1 / self.num_xi_s)
        return density


    def _determine_kernel_shape(self, pattern):
        # Find the K nearest neighbours
        neighbours = self._knn.find_k_nearest_neighbours(pattern, self._k)

        # Compute the covariance matrix
        covariance_matrix = covariance.covariance(neighbours)

        # Compute the scaling factor
        scaling_factor = kde.kernels.scaling.scaling_factor(
            general_bandwidth=self._general_bandwidth,
            covariance_matrix=covariance_matrix
        )

        # Compute the Kernel covariance matrix
        kernel_shape = scaling_factor * covariance_matrix
        return kernel_shape
