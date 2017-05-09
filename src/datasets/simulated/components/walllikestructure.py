from datasets.simulated.components.component import Component


class WallLikeStructure(Component):

    def __init__(self, one_dimensional_components):
        super(WallLikeStructure, self).__init__()
        self._one_dimensional_components = one_dimensional_components

    def patterns(self, num_patterns):
        raise NotImplementedError()

    def densities(self, patterns):
        raise NotImplementedError()