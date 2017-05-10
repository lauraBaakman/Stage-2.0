from collections import OrderedDict

import numpy as np

from inputoutput.dataset import DataSet


class SimulatedDataSet(DataSet):

    def __init__(self):
        np.random.seed(0)
        self._components = OrderedDict()
        self._init_components()
        patterns = self._compute_patterns(self._components)
        densities = self._compute_densities(self._components, patterns)
        super(SimulatedDataSet, self).__init__(
            patterns=patterns,
            densities=densities
        )

    def _init_components(self):
        raise NotImplementedError()

    def _compute_patterns(self, components):
        component_patterns = list(map(
            lambda component:
            component['component'].patterns(component['num elements']), components.values()
        ))

        patterns = np.vstack(component_patterns)
        return patterns

    def _compute_densities(self, components, patterns):
        individual_densities = list(map(
            lambda component:
            component['component'].densities(patterns), components.values()
        ))
        return np.mean(individual_densities, axis=0)

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)
