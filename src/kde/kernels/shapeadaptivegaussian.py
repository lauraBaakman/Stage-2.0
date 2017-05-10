import numpy.linalg as LA
import scipy.stats as stats
import numpy as np

import kde.kernels._kernels as _kernels
from kde.kernels.kernel import Kernel, KernelException

_as_c_enum = 4

_default_local_bandwidth = 1.0

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
        self._global_bandwidth_matrix_inverse = LA.inv(bandwidth_matrix)
        self._scaling_factor = 1 / LA.det(bandwidth_matrix)

    @property
    def dimension(self):
        (dimension, _) = self._global_bandwidth_matrix_inverse.shape
        return dimension

    def evaluate(self, xs, local_bandwidth=None):
        raise NotImplementedError("No functions should be called on objects of this type, it is an abstract class "
                                  "for the specific implementations.")

    def _define_and_validate_input(self, xs_in, local_bandwidths_in):
        xs = self._define_and_validate_patterns(xs_in)
        local_bandwidths = self._define_and_validate_local_bandwidths(local_bandwidths_in, xs)
        return (xs, local_bandwidths)

    def _define_and_validate_patterns(self, xs):
        xs = np.array(xs, ndmin=2)
        xs_dimension = self._get_data_dimension(xs)
        if xs_dimension is not self.dimension:
            raise KernelException("Patterns should have dimension {}, not {}.".format(self.dimension, xs_dimension))
        return xs

    def _define_and_validate_local_bandwidths(self, input_local_bandwidths, xs):
        (num_patterns, _) = xs.shape

        if input_local_bandwidths is None:
            return self._create_default_local_bandwidths_array(num_patterns)
        local_bandwidths = np.array(input_local_bandwidths, ndmin=1)
        self._validate_local_bandwidths(local_bandwidths, num_patterns)
        return local_bandwidths

    def _create_default_local_bandwidths_array(self, num_patterns):
        local_bandwidths = np.empty(num_patterns, dtype=np.float64)
        local_bandwidths.fill(_default_local_bandwidth)
        return local_bandwidths

    def _validate_local_bandwidths(self, local_bandwidths, num_patterns):
        if not (local_bandwidths.ndim == 1):
            raise KernelException(
                "The array with local bandwidths should be 1D, not {}D.".format(local_bandwidths.ndim)
            )
        (num_local_bandwidths,) = local_bandwidths.shape
        if not (num_local_bandwidths == num_patterns):
            raise KernelException(
                "The number of local bandwidths ({}) should be equal to the number of patterns ({}).".format(
                    num_local_bandwidths, num_patterns
                )
            )

    def to_C_enum(self):
        return _as_c_enum


class _ShapeAdaptiveGaussian_C(_ShapeAdaptiveGaussian):
    def __init__(self, *args, **kwargs):
        super(_ShapeAdaptiveGaussian_C, self).__init__(*args, **kwargs)

    def evaluate(self, xs, local_bandwidths=None):
        (xs, local_bandwidths) = self._define_and_validate_input(xs, local_bandwidths)

        if xs.ndim == 1:
            return self._handle_single_pattern(xs)
        elif xs.ndim == 2:
            return self._handle_multiple_patterns(xs)
        else:
            raise TypeError("Expected a vector or a matrix, not a {}-dimensional array.".format(xs.ndim))

    def _handle_single_pattern(self, x):
        data = np.array([x])
        density = _kernels.sa_gaussian_single_pattern(data, self._global_bandwidth_matrix)
        return density

    def _handle_multiple_patterns(self, xs):
        pass


class _ShapeAdaptiveGaussian_Python(_ShapeAdaptiveGaussian):

    def __init__(self, *args, **kwargs):
        super(_ShapeAdaptiveGaussian_Python, self).__init__(*args, **kwargs)
        self._distribution = stats.multivariate_normal(mean=np.zeros([self.dimension]))

    def evaluate(self, xs, local_bandwidths=None):
        (xs, local_bandwidths) = self._define_and_validate_input(xs, local_bandwidths)

        (num_patterns, _) = xs.shape
        densities = np.empty(num_patterns)

        for idx, (pattern, local_bandwidth) in enumerate(zip(xs, local_bandwidths)):
            densities[idx] = self._evaluate_pattern(pattern, local_bandwidth)

        return self._handle_return(densities)

    def _handle_return(self, densities):
        try:
            densities = np.asscalar(densities)
        except ValueError:
            pass #We are dealing with a vector, let's return that
        return densities

    def _evaluate_pattern(self, pattern, local_bandwidth):
        local_inverse = self._compute_local_inverse(local_bandwidth)
        local_scaling_factor = self._compute_local_scaling_factor(local_bandwidth)
        density = self._distribution.pdf(np.matmul(pattern, local_inverse))
        return local_scaling_factor * density

    def _compute_local_scaling_factor(self, local_bandwidth):
        return (1 / np.power(local_bandwidth, self.dimension)) * self._scaling_factor

    def _compute_local_inverse(self, local_bandwidth):
        return (1.0 / local_bandwidth) * self._global_bandwidth_matrix_inverse
