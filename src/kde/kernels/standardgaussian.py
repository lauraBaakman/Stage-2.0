import warnings

import numpy as np
import scipy.stats as stats


class StandardGaussian:

    def __init__(self, dimension=3):
        mean = np.zeros(dimension)
        covariance = np.identity(dimension)
        self._kernel = stats.multivariate_normal(mean=mean, cov=covariance)

    @property
    def center(self):
        warnings.warn(
            "Center property of the Standard Gaussian cannot be retrieved, it is only here for consistency",
            category=UserWarning)

    @center.setter
    def center(self, value):
        warnings.warn(
            "Center property of the Standard Gaussian cannot be changed, it is only here for consistency",
            category=UserWarning)

    @property
    def shape(self):
        warnings.warn(
            "Shape property of the Standard Gaussian cannot be retrieved, it is only here for consistency",
            category=UserWarning)

    @shape.setter
    def shape(self, value):
        warnings.warn(
            "Shape property of the Standard Gaussian cannot be changed, it is only here for consistency",
            category=UserWarning)

    def evaluate(self, x):
        return self._kernel.pdf(x)