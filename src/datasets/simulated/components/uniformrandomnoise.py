import numpy as np
import scipy.stats as stats

from datasets.simulated.components.component import Component


class UniformRandomNoise(Component):

    def __init__(self, minimum_value, maximum_value, dimension=3):
        super(UniformRandomNoise, self).__init__()
        self._min_value = minimum_value
        self._max_value = maximum_value
        self._dimension = dimension

    @property
    def location(self):
        return self._min_value

    @property
    def scale(self):
        return self._max_value - self._min_value

    def patterns(self, num_patterns):
        patterns = np.random.uniform(self._min_value, self._max_value,
                                     size=(num_patterns, self._dimension))
        return patterns

    def densities(self, patterns):
        densities_one_d = stats.uniform.pdf(patterns, loc=self.location, scale=self.scale)

        densities = np.prod(densities_one_d, axis=1)
        return np.array(densities, ndmin=1)

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)
