import numpy as np
import numpy.linalg as LA


def _eig_C(data):
    raise NotImplementedError()


def _eig_Python(data):
    eigen_values, eigen_vectors = LA.eig(data)
    return eigen_values, eigen_vectors


def eigenValuesAndVectors(data, implementation=_eig_Python):
    """
    :param data:
    :param implementation:
    :return: Each column of the matrix with eigenvectors is a eigenvector.
    """
    return implementation(data)