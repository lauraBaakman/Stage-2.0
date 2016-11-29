from enum import Enum


# http://stackoverflow.com/questions/17177061/epanechnikov-multivariate-density
class Multivariate(Enum):
    multiplication = 1
    norm = 2


class Epanechnikov:

    def __init__(self, dimension, multivariate_approach=Multivariate.multiplication):
        self._dimension = dimension
        if dimension is 1:
            self.evaluate = self._univariate
        elif dimension > 1 and multivariate_approach is Multivariate.multiplication:
            self.evaluate = self._multivariate_multiplicative
        elif dimension > 1 and multivariate_approach is Multivariate.norm:
            self.evaluate = self._multivariate_norm

    def _univariate(self, x):
        print("Univariate")
        raise NotImplementedError()

    def _multivariate_multiplicative(self, x):
        print("Multiplicative")
        raise NotImplementedError()

    def _multivariate_norm(self, x):
        print("Norm")
        raise NotImplementedError