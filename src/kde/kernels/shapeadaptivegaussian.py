import numpy.linalg as LA

from kde.kernels.kernel import Kernel, KernelException

_as_c_enum = 4


class ShapeAdaptiveGaussian(object):

    def __new__(cls, bandwidth_matrix, implementation=None):
        implementation_class = implementation or _ShapeAdaptiveGaussian_C
        cls._evaluate_bandwidth_matrix(bandwidth_matrix)
        return implementation_class(bandwidth_matrix=bandwidth_matrix)

    @classmethod
    def _evaluate_bandwidth_matrix(self, bandwidth_matrix):
        raise NotImplementedError()

    @staticmethod
    def to_C_enum():
        return _as_c_enum


class _ShapeAdaptiveGaussian(Kernel):

    def __init__(self, bandwidth_matrix, *args, **kwargs):
        self._bandwidth_matrix = bandwidth_matrix
        self._bandwidth_matrix_inverse = LA.inv(bandwidth_matrix)
        self._bandwidth_matrix_determinant = LA.det(bandwidth_matrix)

    @property
    def dimension(self):
        raise NotImplementedError()

    def evaluate(self, xs, local_bandwidth=1):
        raise NotImplementedError("No functions should be called on objects of this type, it is an abstract class "
                                  "for the specific implementations.")

    def _validate_patterns(self, xs):
        raise NotImplementedError()

    @staticmethod
    def to_C_enum():
        return _as_c_enum


class _ShapeAdaptiveGaussian_C(_ShapeAdaptiveGaussian):
    def __init__(self, *args, **kwargs):
        super(_ShapeAdaptiveGaussian_C, self).__init__(*args, **kwargs)

    def evaluate(self, xs, local_bandwidth=1):
        self._validate_patterns(xs)
        if xs.ndim == 1:
            return self._handle_single_pattern(xs)
        elif xs.ndim == 2:
            return self._handle_multiple_patterns(xs)
        else:
            raise TypeError("Expected a vector or a matrix, not a {}-dimensional array.".format(xs.ndim))

    def _handle_single_pattern(self, x):
        raise NotImplementedError()

    def _handle_multiple_patterns(self, xs):
        raise NotImplementedError()


class _ShapeAdaptiveGaussian_Python(_ShapeAdaptiveGaussian):

    def __init__(self, *args, **kwargs):
        super(_ShapeAdaptiveGaussian_C, self).__init__(*args, **kwargs)

    def evaluate(self, xs, local_bandwidth=1):
        self._validate_patterns(xs)
        raise NotImplementedError()