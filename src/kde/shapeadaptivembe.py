import numpy as np

from kde.modifeidbreiman import ModifiedBreimanEstimator
import kde.utils.automaticWindowWidthMethods as automaticWindowWidthMethods
from kde.estimatorimplementation import EstimatorImplementation
from kde.parzen import ParzenEstimator
from kde.utils.knn import KNN
import kde.utils.covariance as covariance
import kde.utils.eigenvalues as eigenvalues
from kde.kernels.gaussian import Gaussian
from kde.kernels.epanechnikov import Epanechnikov


class ShapeAdaptiveMBE(ModifiedBreimanEstimator):
    """
    Implementation of the shape adaptive modified Breiman Estimator.
    """

    def __init__(self, dimension, sensitivity=1 / 2,
                 pilot_window_width_method=automaticWindowWidthMethods.ferdosi,
                 number_of_grid_points=ModifiedBreimanEstimator.default_number_of_grid_points,
                 pilot_kernel_class=None, pilot_estimator_implementation=None,
                 kernel_class=None, final_estimator_implementation=None):
        # super().__init__(dimension, kernel_class, sensitivity, pilot_kernel_class, pilot_window_width_method, number_of_grid_points,
        #                  pilot_estimator_implementation, final_estimator_implementation)
        self._dimension = dimension
        self._general_window_width_method = pilot_window_width_method
        self._sensitivity = sensitivity
        self._pilot_kernel_class = pilot_kernel_class or Epanechnikov
        self._kernel = kernel_class or Gaussian
        self._number_of_grid_points = number_of_grid_points or ModifiedBreimanEstimator.default_number_of_grid_points

        self._pilot_estimator_implementation = pilot_estimator_implementation or ParzenEstimator
        self._final_estimator_implementation = final_estimator_implementation or _ShapeAdaptiveMBE_C


class _ShapeAdaptiveMBE(EstimatorImplementation):
    def __init__(self, xi_s, x_s, dimension, kernel, local_bandwidths, general_bandwidth):
        super().__init__(xi_s, x_s, dimension, kernel, general_bandwidth)
        self._local_bandwidths = local_bandwidths.astype(float, copy=False)
        self._knn = KNN(patterns=self._xi_s)
        self._k = int(round(max(np.sqrt(self.num_xi_s), self.dimension))) + 1
        self._kernel = None
        self._kernel_class = kernel or Gaussian

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

    def _estimate_pattern(self, pattern):
        kernel_shape = self._determine_kernel_shape(pattern)
        factors = np.power(self._local_bandwidths * self._general_bandwidth, - self._dimension)
        kernel = self._kernel_class(
            mean=np.zeros([self.dimension], dtype=np.float64),
            covariance_matrix=kernel_shape
        )
        terms = kernel.evaluate((pattern - self._xi_s))
        terms *= factors
        density = (1 / self.num_xi_s) * terms.sum()
        return density

    def _determine_kernel_shape(self, pattern):
        # Find the K nearest neighbours
        neighbours = self._knn.find_k_nearest_neighbours(pattern, self._k)

        # Compute the covariance matrix
        covariance_matrix = covariance.covariance(neighbours)

        # Compute the eigenvalues of the covariance matrix
        eigen_values = eigenvalues.eigenvalues(covariance_matrix)

        # Compute the scaling factor
        scaling_factor = self._kernel_class.scaling_factor(
            general_bandwidth=self._general_bandwidth,
            eigen_values=eigen_values
        )

        # Compute the Kernel covariance matrix
        kernel_shape = scaling_factor * covariance_matrix
        return kernel_shape