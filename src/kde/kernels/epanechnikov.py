from enum import Enum

import numpy as np

# http://stackoverflow.com/questions/17177061/epanechnikov-multivariate-density
class Multivariate(Enum):
    multiplication = 1
    norm = 2


def _univariate(x):
    return np.squeeze(3/4 * (1 - np.square(x)) * (abs(x) <= 1))


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

    def __init__(self, dimension, multivariate_approach=Multivariate.multiplication):
        if dimension is 1:
            self.evaluate = _univariate
        elif dimension > 1 and multivariate_approach is Multivariate.multiplication:
            self.evaluate = _multivariate_multiplicative
        elif dimension > 1 and multivariate_approach is Multivariate.norm:
            self.evaluate = _multivariate_norm
