from collections import OrderedDict
from functools import reduce

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

    @property
    def components_lengths(self):
        return [
            component['num elements']
            for component
            in self._components.values()
        ]

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

    def to_file(self, out_file):
        self._header_to_file(out_file)
        self._patterns_to_file(out_file)
        self._densities_to_file(out_file)

    def _header_to_file(self, outfile):
        outfile.write(
            '{length} {dimension}\n'.format(
                length=self.num_patterns,
                dimension=self.dimension
            ).encode('utf-8')
        )
        outfile.write(
            '{component_lengths}\n'.format(
                component_lengths=reduce(lambda x, y: '{} {}'.format(x, y), self.components_lengths)
            ).encode('utf-8')
        )

    def _patterns_to_file(self, outfile):
        np.savetxt(outfile, self.patterns)

    def _densities_to_file(self, outfile):
        np.savetxt(outfile, self.densities)

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)
