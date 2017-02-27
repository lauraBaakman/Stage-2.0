import math

import kde._kde as _kde
import numpy as np

from kde.estimatorimplementation import EstimatorImplementation
from kde.kernels.standardGaussian import StandardGaussian


class ParzenEstimator(object):
    def __init__(self, dimension, bandwidth, kernel_class, estimator_implementation=None):
        self._dimension = dimension
        self._bandwidth = bandwidth
        self._kernel = kernel_class() or StandardGaussian()
        self._estimator_implementation = estimator_implementation or _ParzenEstimator_C

    def estimate(self, xi_s, x_s=None):

        if x_s is None:
            x_s = xi_s
        estimator = self._estimator_implementation(
            xi_s=xi_s, x_s=x_s,
            dimension=self._dimension,
            kernel=self._kernel, general_bandwidth=self._bandwidth
        )
        return estimator.estimate()


class _ParzenEstimator(EstimatorImplementation):
    def __init__(self, xi_s, x_s, dimension, kernel, general_bandwidth):
        super(_ParzenEstimator, self).__init__(
            xi_s=xi_s, x_s=x_s,
            dimension=dimension,
            general_bandwidth=general_bandwidth, kernel=kernel
        )

    def estimate(self):
        raise NotImplementedError("The class_ParzenEstimator is abstract, and should not be called."
                                  "Call one of the classes that inherits from it.")


class _ParzenEstimator_Python(_ParzenEstimator):
    def __init__(self, xi_s, x_s, dimension, kernel, general_bandwidth):
        super(_ParzenEstimator_Python, self).__init__(
            xi_s=xi_s, x_s=x_s,
            dimension=dimension,
            kernel=kernel, general_bandwidth=general_bandwidth
        )

    def estimate(self):
        self._kernel.center = np.zeros(self._dimension)
        self._kernel.shape = np.identity(self._dimension)
        densities = np.empty(self.num_x_s)
        factor = 1 / (self.num_xi_s * math.pow(self._general_bandwidth, self._dimension))
        for idx, x in enumerate(self._x_s):
            densities[idx] = self._estimate_pattern(x, factor)
        return densities

    def _estimate_pattern(self, pattern, factor):
        terms = self._kernel.evaluate((pattern - self._xi_s) / self._general_bandwidth)
        density = factor * terms.sum()
        return density


class _ParzenEstimator_C(_ParzenEstimator):
    def __init__(self, xi_s, x_s, dimension, kernel, general_bandwidth):
        super(_ParzenEstimator_C, self).__init__(
            xi_s=xi_s, x_s=x_s,
            dimension=dimension,
            kernel=kernel, general_bandwidth=general_bandwidth
        )

    def estimate(self):
        densities = np.empty(self.num_x_s, dtype=float)
        _kde.parzen(self._x_s, self._xi_s, self._general_bandwidth, self._kernel.to_C_enum(), densities)
        return densities
