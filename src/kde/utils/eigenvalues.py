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
    _order_greater_than_two(data)


def _has_two_dimensions(data):
    if data.ndim is not 2:
        raise ValueError("Expected an array with two dimensions, not {}.".format(data.ndim))


def _is_square(data):
    (num_rows, num_cols) = data.shape
    if num_rows is not num_cols:
        raise ValueError("Expected a square matrix, not a {} x {} matrix.".format(num_rows, num_cols))


def _order_greater_than_two(data):
    (num_rows, num_cols) = data.shape
    if num_rows < 2 or num_cols < 2:
        raise ValueError("The matrix needs to be at least 2 x 2, the input matrix is {} x {}.".format(num_rows, num_cols))


def _eigenvalues_Python(data):
    eigen_values, _ = LA.eig(data)
    return eigen_values


def eigenvalues(data, implementation=_eigenvalues_Python):
    """
    :param data:
    :param implementation:
    :return: Array with eigenvalues.
    """
    return implementation(data)
