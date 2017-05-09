import numpy as np
import scipy.stats as stats

from datasets.simulated.components.component import Component


class MultivariateGaussian(Component):
    def __init__(self, mean, covariance_matrix):
        super(MultivariateGaussian, self).__init__()
        self._mean = mean
        self._covariance_matrix = covariance_matrix

    def patterns(self, num_patterns):
        patterns = np.random.multivariate_normal(
            self._mean,
            self._covariance_matrix,
            num_patterns)
        return patterns

    def densities(self, patterns):
        densities = stats.multivariate_normal.pdf(
            x=patterns,
            mean=self._mean,
            cov=self._covariance_matrix)
        return np.array(densities, ndmin=1)

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)


class UnivariateGaussian(Component):
    def __init__(self, mean, variance):
        super(UnivariateGaussian, self).__init__()
        self._mean = mean
        self._variance = variance

    @property
    def standard_deviation(self):
        return np.sqrt(self._variance)

    def patterns(self, num_patterns):
        patterns = np.random.normal(
            self._mean,
            self.standard_deviation,
            num_patterns)
        return np.array([np.array(x, ndmin=1) for x in patterns])

    def densities(self, patterns):
        densities = stats.norm.pdf(patterns, self._mean, self.standard_deviation)
        return np.array(np.squeeze(densities), ndmin=1)
