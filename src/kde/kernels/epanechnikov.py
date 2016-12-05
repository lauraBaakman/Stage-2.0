from enum import Enum

import numpy as np



class Multivariate(Enum):
    """Enum to indicate how to evaluate the multivariate Epanechnikov Kernel. Possible options:
        Multiplication: Evaluate the 1D kernel for each dimension and multiply the results. Assumes that the dimensions
            are statistically independent.
        Norm: Evaluate the 1D kernel for each dimension and store the results in a vector, the evaluation of the
            multivariate Epanechnikov kernel is the norm of this vector.

        Source: # http://stackoverflow.com/questions/17177061/epanechnikov-multivariate-density
    """
    multiplication = 1
    norm = 2


def _univariate(x):
    return np.squeeze(3 / 4 * (1 - np.square(x)) * (abs(x) <= 1))


def _multivariate_multiplicative(x):
    univariate_results = _univariate(x)
    try:
        value = np.prod(univariate_results, axis=1)
    except ValueError:
        value = np.prod(univariate_results)
    return value


def _multivariate_norm(x):
    univariate_results = _univariate(x)
    try:
        value = np.linalg.norm(univariate_results, axis=1)
    except ValueError:
        value = np.linalg.norm(univariate_results)
    return value


class Epanechnikov:
    """Implementation of the Epanechnikov Kernel.
    """

    def __init__(self, dimension, multivariate_approach=Multivariate.multiplication):
        """ Init method of the Epanechnikov Kernel.
        :param dimension (int): The dimension of the kernel.
        :param multivariate_approach (Multivariate): How to evaluate the multivariate kernel. Defaults to None.
        """
        if dimension is 1:
            self.evaluate = _univariate
        elif dimension > 1 and multivariate_approach is Multivariate.multiplication:
            self.evaluate = _multivariate_multiplicative
        elif dimension > 1 and multivariate_approach is Multivariate.norm:
            self.evaluate = _multivariate_norm
