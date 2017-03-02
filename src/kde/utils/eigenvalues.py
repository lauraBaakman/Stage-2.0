import numpy as np
import numpy.linalg as LA

import kde.utils._utils as _utils


def _eigenvalues_C(data):
    _validate_input_matrix(data)

    # Allocate array for the eigenvalues
    # Call the C function
    raise NotImplementedError("There is no C implementation of the computation of eigen values.")


def _validate_input_matrix(data):
    _has_two_dimensions(data)
    _is_square(data)
    _is_at_least_2_times_2(data)


def _is_square(data):
    raise NotImplementedError()


def _has_two_dimensions(data):
    raise NotImplementedError()


def _is_at_least_2_times_2(data):
    raise NotImplementedError()


def _eigenvalues_Python(data):
    eigen_values, _ = LA.eig(data)
    return eigen_values


def eigenvalues(data, implementation=_eigenvalues_Python):
    """
    :param data:
    :param implementation:
    :return: Each column of the matrix with eigenvectors is a eigenvector.
    """
    return implementation(data)
