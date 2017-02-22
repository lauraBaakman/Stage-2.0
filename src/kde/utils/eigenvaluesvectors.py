import numpy as np


def _eig_C(data):
    raise NotImplementedError()


def _eig_Python(data):
    raise NotImplementedError()


def eigenValuesAndVectors(data, implementation=_eig_Python):
    """
    :param data:
    :param implementation:
    :return: Each column of the matrix with eigenvectors is a eigenvector.
    """
    return implementation(data)