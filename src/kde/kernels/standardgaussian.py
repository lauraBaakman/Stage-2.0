import warnings

import numpy as np
import scipy.stats as stats

import _standardGaussian

class StandardGaussian:
    """Implementation of the Standard Gaussian Kernel, i.e. a mean 0 and I as covariance matrix.
    """

    def __init__(self, dimension=3):
        """The init of the Standard Gaussian Kernel.

        :param dimension: (int) The dimension of the kernel.
        """
        mean = np.zeros(dimension)
        covariance = np.identity(dimension)
        self._kernel = stats.multivariate_normal(mean=mean, cov=covariance)

    def evaluate(self, x):
        """Evaluate the kernel for vector x.

        :param x: (array like) The vector for which to evaluate the kernel.
        :return: (int) The probability density of the kernel at x.
        """
        return self._kernel.pdf(x)