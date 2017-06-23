import kde.kernels._kernels as _kernels
import numpy as np

from kde.kernels.kernel import SymmetricKernel

_as_c_enum = 0


class TestKernel(SymmetricKernel):

    @staticmethod
    def __new__(cls, implementation=None):
        implementation_class = implementation or _TestKernel_C
        return implementation_class()

    @staticmethod
    def to_C_enum():
        return _as_c_enum


class _TestKernel(SymmetricKernel):

    def __init__(self):
        pass

    @staticmethod
    def radius(bandwidth):
        return bandwidth * 5

    @staticmethod
    def to_C_enum():
        return _as_c_enum


class _TestKernel_Python(_TestKernel):
    def __init__(self):
        super(_TestKernel_Python, self).__init__()

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


class _TestKernel_C(_TestKernel):
    def __init__(self):
        super(_TestKernel_C, self).__init__()

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
