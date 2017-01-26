import math

import numpy as np
import scipy.special

class Epanechnikov:
    """Implementation of the Epanechnikov Kernel.
    """

    def __init__(self):
        pass

    def evaluate(self, x):
        if x.ndim == 1:
            density = self._evaluate_single_pattern(x)
            return np.array([density])
        elif x.ndim == 2:
            return self._evaluate_multiple_patterns(x)
        else:
            raise TypeError("Expected a vector or a matrix, not a {}-dimensional array.".format(xs.ndim))

    def _evaluate_single_pattern(self, pattern):
        dimension = pattern.size
        x_dot_product = np.dot(pattern, pattern)
        if x_dot_product >= 1:
            return 0
        return (dimension + 2) / (2 * self._unit_sphere_volume(dimension)) * (1 - x_dot_product)

    def _evaluate_multiple_patterns(self, patterns):
        def evaluate(pattern, dimension, volume):
            dot_product = np.dot(pattern, pattern)
            if dot_product >= 1:
                return 0
            return (dimension + 2) / (2 * volume) * (1 - dot_product)

        (num_patterns, dimension) = patterns.shape
        volume = self._unit_sphere_volume(dimension)
        densities = np.empty(num_patterns)
        for idx, pattern in enumerate(patterns):
            densities[idx] = evaluate(pattern=pattern, dimension=dimension, volume=volume)
        return densities


    def _unit_sphere_volume(self, dimension):
        numerator = math.pow(math.pi, dimension / 2.0)
        denominator = scipy.special.gamma(dimension / 2.0 + 1)
        return numerator / denominator