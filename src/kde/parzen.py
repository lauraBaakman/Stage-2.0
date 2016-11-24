import numpy as np
import kde.kernels as kernels


class Parzen:

    def __init__(self, kernel):
        self._kernel = kernel

    def estimate_python(self, data, patterns_to_estimate=None):
        if patterns_to_estimate is None:
            patterns_to_estimate = data
        densities = np.apply_along_axis(
            self._estimate_python_pattern_density, 1, patterns_to_estimate, data)
        return densities

    def _estimate_python_pattern_density(self, x, xi_s):
        self._kernel.center = x
        terms = np.apply_along_axis(lambda x_i: self._kernel.evaluate(x_i), 1, xi_s)
        density = terms.sum()
        return density

    def estimate_pyton_vectorized(self, data, patterns_to_estimate=None):
        if patterns_to_estimate is None:
            patterns_to_estimate = data
        return None


def benchmark_python(n=1000, dimension=3):
    patterns = np.random.randn(n, dimension)
    kernel = kernels.Gaussian(covariance_matrix=0.5 * np.identity(3))
    estimator = Parzen(kernel=kernel)
    densities = estimator.estimate_python(data=patterns)