import numpy as np

import kde.utils.eigenvalues as eig
import kde.kernels._kernels as _kernels


def _scaling_factor_python(local_bandwidth, general_bandwidth, covariance_matrix):
    eigen_values = eig.eigenvalues(covariance_matrix)
    (dimension, _) = covariance_matrix.shape
    bandwidth_term = np.log(local_bandwidth * general_bandwidth)
    eigen_value_term = (1.0 / dimension) * np.sum(np.log(eigen_values))
    return np.exp(bandwidth_term - eigen_value_term)


def _scaling_factor_c(local_bandwidth, general_bandwidth, covariance_matrix):
    return _kernels.scaling_factor(local_bandwidth, general_bandwidth, covariance_matrix)


def scaling_factor(local_bandwidth, general_bandwidth, covariance_matrix, implementation=_scaling_factor_python):
    return implementation(local_bandwidth, general_bandwidth, covariance_matrix)
