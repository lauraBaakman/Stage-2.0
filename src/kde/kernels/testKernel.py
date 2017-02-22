import kde.kernels._kernels as _kernels
import numpy as np

from kde.kernels.kernel import Kernel


class TestKernel(Kernel):
    def __init__(self, implementation=None):
        implementation_class = implementation or _TestKernel_C
        self._implementation = implementation_class()

    def to_C_enum(self):
        return 0


class _TestKernel_Python(Kernel):
    def __init__(self):
        pass

    def evaluate(self, x):
        if x.ndim == 1:
            density = self._evaluate_single_pattern(x)
        elif x.ndim == 2:
            density = self._evaluate_multiple_patterns(x)
        else:
            raise TypeError("Expected a vector or a matrix, not a {}-dimensional array.".format(x.ndim))
        return density

    def _evaluate_single_pattern(self, x):
        return np.abs(np.mean(x))

    def _evaluate_multiple_patterns(self, x):
        return np.abs(np.mean(x, axis=1))

    def scaling_factor(self, general_bandwidth, eigen_values):
        raise NotImplementedError("This class does not have an implementation of the scaling factor computation method.")


class _TestKernel_C(Kernel):
    def __init__(self):
        pass

    def evaluate(self, x):
        if x.ndim == 1:
            data = np.array([x])
            density = _kernels.test_kernel_single_pattern(data)
            return np.array([density])
        elif x.ndim == 2:
            (num_patterns, _) = x.shape
            densities = np.empty(num_patterns, dtype=float)
            _kernels.test_kernel_multi_pattern(x, densities)
            return densities
        else:
            raise TypeError("Expected a vector or a matrix, not a {}-dimensional array.".format(x.ndim))

    def scaling_factor(self, general_bandwidth, eigen_values):
        raise NotImplementedError("This class does not have an implementation of the scaling factor computation method.")
