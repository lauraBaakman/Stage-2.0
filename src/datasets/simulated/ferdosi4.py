import numpy as np

from datasets.simulated.simulateddataset import SimulatedDataSet
import datasets.simulated.components as components


class Ferdosi4(SimulatedDataSet):

    def __init__(self):
        super(Ferdosi4, self).__init__()

    def _init_components(self):
        self._components['wall-like structure'] = {
                'component': components.WallLikeStructure(
                    one_dimensional_components=[
                        components.UniformRandomNoise(0, 100, dimension=1),
                        components.UniformRandomNoise(0, 100, dimension=1),
                        components.UnivariateGaussian(50, 5),
                    ]
                ),
                'num elements': 30000,
            }
        self._components['filament-like structure'] = {
                'component': components.FilamentLikeStructure(
                    one_dimensional_components=[
                        components.UnivariateGaussian(50, 5),
                        components.UnivariateGaussian(50, 5),
                        components.UniformRandomNoise(0, 100, dimension=1),
                    ]
                ),
                'num elements': 30000,
            }
