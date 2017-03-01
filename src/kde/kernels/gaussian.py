import scipy.stats as stats
from scipy.stats.mstats import gmean
import numpy as np

from kde.kernels.kernel import Kernel, KernelException


_as_c_enum = 3


def _scaling_factor(general_bandwidth, eigen_values):
    return (general_bandwidth * general_bandwidth) / gmean(np.power(eigen_values, (1.0 / 2)))


class Gaussian(object):

    def __new__(cls, mean, covariance_matrix, implementation=None):
        implementation_class = implementation or _Gaussian_Python
        return implementation_class(mean=mean, covariance_matrix=covariance_matrix)

    @staticmethod
    def scaling_factor(general_bandwidth, eigen_values):
        return _scaling_factor(general_bandwidth, eigen_values)

    @staticmethod
    def to_C_enum():
        return _as_c_enum


class _Gaussian(Kernel):

    def __init__(self, mean, covariance_matrix, *args, **kwargs):
        self._mean = mean
        self._covariance_matrix = covariance_matrix
        self._validate_parameters(mean, covariance_matrix)

    @property
    def dimension(self):
        (dimension, _) = self._covariance_matrix.shape
        return dimension

    def _validate_mean_covariance_combination(self, mean, covariance_matrix):
        (covariance_dimension, _) = covariance_matrix.shape
        mean_dimension = self._get_data_dimension(mean)
        if mean_dimension is not covariance_dimension:
            raise KernelException("If the covariance matrix is {} x {}, the mean should be 1 x {}".format(
                covariance_dimension, covariance_dimension, mean_dimension
            ))

    def _validate_xs_pdf_combination(self, xs):
        xs_dimension = self._get_data_dimension(xs)
        if xs_dimension is not self.dimension:
            raise KernelException("Patterns should have dimension {}, not {}.".format(self.dimension, xs_dimension))

    def evaluate(self, xs):
        raise NotImplementedError("No functions should be called on objects of this type, it is an abstract class "
                                  "for the specific implementations.")

    @staticmethod
    def to_C_enum(self):
        return _as_c_enum

    def _validate_parameters(self, mean, covariance_matrix):
        self._validate_mean(mean)
        self._validate_covariance_matrix(covariance_matrix)
        self._validate_mean_covariance_combination(mean, covariance_matrix)

    def _validate_mean(self, mean):
        if mean.ndim is not 1:
            raise KernelException("The array with the mean should have 1 dimension, not {}.".format(mean.ndim))

    def _validate_covariance_matrix(self, covariance_matrix):
        if covariance_matrix.ndim is not 2:
            raise KernelException("The covariance matrix should have 2 dimensions, not {}.".format(covariance_matrix.ndim))

        (num_rows, num_cols) = covariance_matrix.shape
        if num_rows is not num_cols:
            raise KernelException("The covariance matrix should be square.")


class _Gaussian_C(_Gaussian):
    def __init__(self, *args, **kwargs):
        super(_Gaussian_C, self).__init__(*args, **kwargs)

    def evaluate(self, xs):
        self._validate_xs_pdf_combination(xs)
        raise NotImplementedError("Gaussian kernel is not yet implemented in C.")

    @staticmethod
    def scaling_factor(general_bandwidth, eigen_values):
        raise NotImplementedError("Gaussian kernel is not yet implemented in C.")


class _Gaussian_Python(_Gaussian):
    def __init__(self, *args, **kwargs):
        super(_Gaussian_Python, self).__init__(*args, **kwargs)
        try:
            self._kernel = stats.multivariate_normal(mean=self._mean, cov=self._covariance_matrix)
        except ValueError as e:
            raise KernelException("Could not generate the multivariate normal, numpy error: {}".format(e.args[0]))

    def evaluate(self, xs):
        self._validate_xs_pdf_combination(xs)
        return self._kernel.pdf(xs)

    @staticmethod
    def scaling_factor(general_bandwidth, eigen_values):
        return _scaling_factor(general_bandwidth, eigen_values)
