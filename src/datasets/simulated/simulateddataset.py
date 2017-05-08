from inputoutput.dataset import DataSet


class SimulatedDataSet(DataSet):

    def __init__(self):
        self._components = self._init_components()
        patterns = self._compute_patterns()
        densities = self._compute_densities(patterns)
        super(SimulatedDataSet, self).__init__(
            patterns=patterns,
            densities=densities
        )

    def _init_components(self):
        raise NotImplementedError()

    @property
    def number_of_elements(self):
        num_elements = 0
        for component in self._components.values:
            num_elements += component['num elements']
        return num_elements

    def _compute_patterns(self):
        for component in self._components.values():
            component_patterns = component['component'].patterns(component['num elements'])
        raise NotImplementedError

    def _compute_densities(self, patterns):
        raise NotImplementedError()

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)
