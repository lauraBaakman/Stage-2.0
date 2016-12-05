import scipy.stats as stats


class Gaussian:
    """Implementation of the Gaussian Kernel.
    """

    def __init__(self, mean=None, covariance_matrix=None):
        """The init method of the Gaussian Kernel.

        Note:
            Although the mean and covariance are optional, not setting them means that the kernel cannot be evaluated.

        :param mean: (array like) The mean of the kernel.
        :param covariance_matrix: (array like) The covariance matrix of the kernel.
        """
        self._covariance_matrix = covariance_matrix
        self._mean = mean
        self._kernel = self._updated_kernel()

    @property
    def center(self):
        """
        :return: (array-like) The center, i.e. the mean, of the Gaussian kernel.
        """
        return self._mean

    @center.setter
    def center(self, value):
        self._mean = value
        self._kernel = self._updated_kernel()

    @property
    def shape(self):
        """
        :return: (array-like) The covariance matrix of the kernel.
        """
        return self._covariance_matrix

    @shape.setter
    def shape(self, value):
        self._covariance_matrix = value
        self._kernel = self._updated_kernel()

    def _updated_kernel(self):
        if (self._mean is not None) and (self._covariance_matrix is not None):
            return stats.multivariate_normal(
                mean=self._mean,
                cov=self._covariance_matrix
            )
        else:
            return None

    def evaluate(self, x):
        """Evaluate the kernel for vector x.

        :param x: (array like) The vector for which to evaluate the kernel.
        :return: (int) The probability density of the kernel at x.
        """
        return self._kernel.pdf(x)