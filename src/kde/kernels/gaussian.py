import scipy.stats as stats


class Gaussian:

    def __init__(self, mean=None, covariance_matrix=None):
        self._covariance_matrix = covariance_matrix
        self._mean = mean
        self._update_kernel()

    @property
    def center(self):
        return self._mean

    @center.setter
    def center(self, value):
        self._mean = value
        self._update_kernel()

    @property
    def shape(self):
        return self._covariance_matrix

    @shape.setter
    def shape(self, value):
        self._covariance_matrix = value
        self._update_kernel()

    def _update_kernel(self):
        if (self._mean is not None) and (self._covariance_matrix is not None):
            self._kernel = stats.multivariate_normal(
                mean=self._mean,
                cov=self._covariance_matrix
            )
        else:
            self._kernel = None

    def evaluate(self, x):
        return self._kernel.pdf(x)