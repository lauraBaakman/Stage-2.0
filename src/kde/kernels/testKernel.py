import numpy as np

from kde.kernels.kernel import Kernel


class TestKernel(Kernel):
    def __init__(self, implementation=None):
        implementation_class = implementation or _TestKernel_Python
        self._implementation = implementation_class()

    def to_C_enum(self):
        return 3


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


class _TestKernel_C(Kernel):
    def __init__(self):
        pass

    def evaluate(self, x):
        raise NotImplementedError()
