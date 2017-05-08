import numpy as np
import scipy.stats as stats

from datasets.simulated.components.component import Component


class TrivariateGaussian(Component):

    def __init__(self, mean, covariance_matrix):
        super(TrivariateGaussian, self).__init__()
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
        return densities

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)