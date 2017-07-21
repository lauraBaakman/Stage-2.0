from __future__ import division

import os
import math

import kde._kde as _kde
import numpy as np

from inputoutput.results import Results
import kde.utils.automaticWindowWidthMethods as automaticWindowWidthMethods
from kde.estimatorimplementation import EstimatorImplementation
from kde.kernels.gaussian import Gaussian


class ParzenEstimator(object):
    def __init__(self, dimension, bandwidth=None, kernel_class=None, estimator_implementation=None, *args, **kwargs):
        self._dimension = dimension
        self._bandwidth = bandwidth
        self._kernel = kernel_class() if kernel_class else Gaussian()
        self._estimator_implementation = estimator_implementation or _ParzenEstimator_C

    def estimate(self, xi_s, x_s=None, general_bandwidth=None, *args, **kwargs):

        if x_s is None:
            x_s = xi_s
        if os.environ.get('DEBUGOUTPUT'):
            print('\t\t\tEstimating {} densities'.format(x_s.shape[0]))

        if general_bandwidth is None:
            if self._bandwidth is None:
                self._bandwidth = automaticWindowWidthMethods.ferdosi(xi_s)
        else:
            self._bandwidth = general_bandwidth

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
        factor = 1 / (self.num_xi_s * math.pow(self._general_bandwidth, self._dimension))
        results = Results(expected_size=self.num_x_s)
        for idx, x in enumerate(self._x_s):
            density = self._estimate_pattern(x, factor)
            results.add_result(density=density)
        return results

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
        result = Results(densities=densities)
        return result
