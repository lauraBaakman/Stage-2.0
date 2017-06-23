import numpy as np
from numpy import linalg as LA

default_local_bandwidth = 1.0


class Kernel(object):
    def __init__(self):
        pass

    def evaluate(self, xs):
        raise NotImplementedError()

    def to_C_enum(self):
        raise NotImplementedError()

    def _get_data_dimension(self, xs):
        if xs.ndim == 1:
            (dimension,) = xs.shape
        elif xs.ndim == 2:
            (_, dimension) = xs.shape
        else:
            raise TypeError("Expected a vector or a matrix, not a {}-dimensional array.".format(xs.ndim))
        return dimension


class SymmetricKernel(Kernel):
    def __init__(self):
        pass

    def radius(bandwidth):
        raise NotImplementedError()


class ShapeAdaptive(object):

    @staticmethod
    def to_C_enum():
        raise NotImplementedError()

    @classmethod
    def _validate_bandwidth_matrix(cls, bandwidth_matrix):
        def _is_square(matrix):
            (rows, cols) = matrix.shape
            return rows == cols

        def _is_2D(matrix):
            return matrix.ndim == 2

        if not _is_2D(bandwidth_matrix):
            raise KernelException("The bandwidth matrix should be 2D.")
        if not _is_square(bandwidth_matrix):
            raise KernelException("The bandwidth matrix should be square.")


class ShapeAdaptiveKernel(Kernel):

    def __init__(self, bandwidth_matrix, *args, **kwargs):
        super(ShapeAdaptiveKernel, self).__init__()
        self._global_bandwidth_matrix = bandwidth_matrix

    def evaluate(self, xs):
        raise NotImplementedError()

    def to_C_enum(self):
        raise NotImplementedError()

    def _define_and_validate_input(self, xs_in, local_bandwidths_in):
        xs = self._define_and_validate_patterns(xs_in)
        local_bandwidths = self._define_and_validate_local_bandwidths(local_bandwidths_in, xs)
        return xs, local_bandwidths

    def _define_and_validate_patterns(self, xs):
        xs = np.array(xs, ndmin=2)
        if xs.ndim is not 2:
            raise KernelException("Expected a vector or a matrix, not a {}-dimensional array.".format(xs.ndim))
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
        local_bandwidths.fill(default_local_bandwidth)
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

    @property
    def dimension(self):
        (dimension, _) = self._global_bandwidth_matrix.shape
        return dimension


class ShapeAdaptiveKernel_C(ShapeAdaptiveKernel):

    def to_C_enum(self):
        pass

    def evaluate(self, xs, local_bandwidths=None):
        (xs, local_bandwidths) = self._define_and_validate_input(xs, local_bandwidths)

        (num_patterns, _) = xs.shape

        if num_patterns == 1:
            return self._handle_single_pattern(xs, local_bandwidths)
        else:
            return self._handle_multiple_patterns(xs, local_bandwidths)

    def _handle_single_pattern(self, x, local_bandwidth):
        pass

    def _handle_multiple_patterns(self, xs, local_bandwidths):
        pass


class ShapeAdaptiveKernel_Python(ShapeAdaptiveKernel):

    def __init__(self, bandwidth_matrix, *args, **kwargs):
        super(ShapeAdaptiveKernel_Python, self).__init__(bandwidth_matrix, *args, **kwargs)
        self._global_bandwidth_matrix_inverse = LA.inv(bandwidth_matrix)
        self._scaling_factor = 1 / LA.det(bandwidth_matrix)

    def to_C_enum(self):
        pass

    def _evaluate_pattern(self, pattern, local_bandwidth):
        pass

    def _handle_return(self, densities):
        try:
            densities = np.asscalar(densities)
        except ValueError:
            pass #We are dealing with a vector, let's return that
        return densities

    def evaluate(self, xs, local_bandwidths=None):
        (xs, local_bandwidths) = self._define_and_validate_input(xs, local_bandwidths)

        (num_patterns, _) = xs.shape
        densities = np.empty(num_patterns)

        for idx, (pattern, local_bandwidth) in enumerate(zip(xs, local_bandwidths)):
            densities[idx] = self._evaluate_pattern(pattern, local_bandwidth)

        return self._handle_return(densities)

    def _compute_local_scaling_factor(self, local_bandwidth):
        return (1 / np.power(local_bandwidth, self.dimension)) * self._scaling_factor

    def _compute_local_inverse(self, local_bandwidth):
        return (1.0 / local_bandwidth) * self._global_bandwidth_matrix_inverse


class KernelException(Exception):

    def __init__(self, message, *args):
        super(KernelException, self).__init__(message, *args)
