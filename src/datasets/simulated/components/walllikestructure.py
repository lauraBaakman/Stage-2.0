import numpy as np

from datasets.simulated.components.component import Component


class WallLikeStructure(Component):

    def __init__(self, one_dimensional_components):
        super(WallLikeStructure, self).__init__()
        self._one_dimensional_components = one_dimensional_components

    def patterns(self, num_patterns):
        patterns = list(
            map(
                lambda component: component.patterns(num_patterns), self._one_dimensional_components
            )
        )
        return np.hstack(patterns)

    def densities(self, patterns):
        columns = [
            np.array([np.array(element, ndmin=1)
                      for element
                      in column])
            for column
            in
            patterns.transpose()
        ]
        column_component_pairs = zip(columns, self._one_dimensional_components)

        one_dimensional_densities = list()
        for (column, component) in column_component_pairs:
            column_densities = component.densities(column)
            one_dimensional_densities.append(column_densities)

        densities = np.prod(one_dimensional_densities, axis=0)
        return densities
