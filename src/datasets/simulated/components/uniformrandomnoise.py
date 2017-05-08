import numpy as np
import scipy.stats as stats

from datasets.simulated.components.component import Component


class UniformRandomNoise(Component):

    def __init__(self, minimum_value, maximum_value):
        super(UniformRandomNoise, self).__init__()
        self._min_value = minimum_value
        self._max_value = maximum_value

    @property
    def location(self):
        return self._min_value

    @property
    def scale(self):
        return self._max_value - self._min_value

    def patterns(self, num_patterns):
        x = np.random.uniform(self._min_value, self._max_value, num_patterns)
        y = np.random.uniform(self._min_value, self._max_value, num_patterns)
        z = np.random.uniform(self._min_value, self._max_value, num_patterns)

        patterns = np.stack((x, y, z), 1)
        return patterns

    def densities(self, patterns):
        densities_1D = stats.uniform.pdf(patterns, loc=self.location, scale=self.scale)

        densities = np.prod(densities_1D, axis=1)
        return np.array(densities, ndmin=1)

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)