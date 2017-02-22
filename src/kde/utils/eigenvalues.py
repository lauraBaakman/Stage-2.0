import numpy as np
import numpy.linalg as LA


def _eigenvalues_C(data):
    raise NotImplementedError("There is no C implementation of the computation of eigen values.")


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
