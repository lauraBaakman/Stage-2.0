import numpy as np

import kde.utils._utils as _utils


def _covariance_C(data):
    (_, dimension) = data.shape
    covariance_matrix = np.empty([dimension, dimension], dtype=np.float64)
    _utils.covariance_matrix(data, covariance_matrix)
    return covariance_matrix


def _covariance_Python(data):
    return np.cov(data.transpose())


def covariance(data, implementation=_covariance_C):
    return implementation(data)

