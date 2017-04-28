import numpy.linalg as LA
import scipy.stats as stats
import numpy as np

from kde.kernels.kernel import Kernel, KernelException

_as_c_enum = 4


class ShapeAdaptiveGaussian(object):

    def __new__(cls, bandwidth_matrix, implementation=None):
        implementation_class = implementation or _ShapeAdaptiveGaussian_Python
        cls._validate_bandwidth_matrix(bandwidth_matrix)
        return implementation_class(bandwidth_matrix=bandwidth_matrix)

    @classmethod
    def _validate_bandwidth_matrix(self, bandwidth_matrix):
        def _is_square(matrix):
            (rows, cols) = matrix.shape
            return rows == cols

        def _is_2D(matrix):
            return matrix.ndim == 2

        if not _is_2D(bandwidth_matrix):
            raise KernelException("The bandwidth matrix should be 2D.")
        if not _is_square(bandwidth_matrix):
            raise KernelException("The bandwidth matrix should be square.")

    @staticmethod
    def to_C_enum():
        return _as_c_enum


class _ShapeAdaptiveGaussian(Kernel):

    def __init__(self, bandwidth_matrix, *args, **kwargs):
        super(_ShapeAdaptiveGaussian, self).__init__()
        self._bandwidth_matrix = bandwidth_matrix
        self._bandwidth_matrix_inverse = LA.inv(bandwidth_matrix)
        self._bandwidth_matrix_determinant = LA.det(bandwidth_matrix)

    @property
    def dimension(self):
        (dimension, _) = self._bandwidth_matrix.shape
        return dimension

    def evaluate(self, xs, local_bandwidth=1):
        raise NotImplementedError("No functions should be called on objects of this type, it is an abstract class "
                                  "for the specific implementations.")

    def _validate_patterns(self, xs):
        xs_dimension = self._get_data_dimension(xs)
        if xs_dimension is not self.dimension:
            raise KernelException("Patterns should have dimension {}, not {}.".format(self.dimension, xs_dimension))

    def to_C_enum(self):
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
        super(_ShapeAdaptiveGaussian_Python, self).__init__(*args, **kwargs)

    def evaluate(self, xs, local_bandwidth=1):
        self._validate_patterns(xs)
        scaling_factor = 1.0 / self._bandwidth_matrix_determinant
        distribution = stats.multivariate_normal(mean=np.zeros([self.dimension]))
        normalized_xs = np.matmul(xs, self._bandwidth_matrix_inverse)
        density = distribution.pdf(normalized_xs)
        return scaling_factor * density
