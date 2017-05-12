import kde.kernels._kernels as _kernels
import numpy as np
import scipy.stats as stats

from kde.kernels.kernel import ShapeAdaptiveKernel_C, ShapeAdaptiveKernel_Python, ShapeAdaptive

_as_c_enum = 4


class ShapeAdaptiveGaussian(ShapeAdaptive):

    def __new__(cls, bandwidth_matrix, implementation=None):
        implementation_class = implementation or _ShapeAdaptiveGaussian_Python
        cls._validate_bandwidth_matrix(bandwidth_matrix)
        return implementation_class(bandwidth_matrix=bandwidth_matrix)

    @staticmethod
    def to_C_enum():
        return _as_c_enum


class _ShapeAdaptiveGaussian_C(ShapeAdaptiveKernel_C):
    def __init__(self, *args, **kwargs):
        super(_ShapeAdaptiveGaussian_C, self).__init__(*args, **kwargs)

    def to_C_enum(self):
        return _as_c_enum

    def _handle_single_pattern(self, x, local_bandwidth):
        data = np.array(x, ndmin=2)
        density = _kernels.sa_gaussian_single_pattern(data, local_bandwidth, self._global_bandwidth_matrix)
        return density

    def _handle_multiple_patterns(self, xs, local_bandwidths):
        (num_patterns, _) = xs.shape
        densities = np.empty(num_patterns, dtype=float)
        _kernels.sa_gaussian_multi_pattern(xs, local_bandwidths, self._global_bandwidth_matrix, densities)
        return densities


class _ShapeAdaptiveGaussian_Python(ShapeAdaptiveKernel_Python):

    def __init__(self, *args, **kwargs):
        super(_ShapeAdaptiveGaussian_Python, self).__init__(*args, **kwargs)
        self._distribution = stats.multivariate_normal(mean=np.zeros([self.dimension]))

    def to_C_enum(self):
        return _as_c_enum

    def _evaluate_pattern(self, pattern, local_bandwidth):
        local_inverse = self._compute_local_inverse(local_bandwidth)
        local_scaling_factor = self._compute_local_scaling_factor(local_bandwidth)
        density = self._distribution.pdf(np.matmul(pattern, local_inverse))
        return local_scaling_factor * density
