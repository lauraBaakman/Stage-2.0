import numpy as np

from datasets.simulated.simulateddataset import SimulatedDataSet
import datasets.simulated.components as components


class Ferdosi4(SimulatedDataSet):

    def __init__(self, scale=1.0):
        super(Ferdosi4, self).__init__(scale)

    def _init_components(self):
        self._components['wall-like structure'] = {
                'component': components.WallLikeStructure(
                    one_dimensional_components=[
                        components.UniformRandomNoise(0, 100, dimension=1),
                        components.UniformRandomNoise(0, 100, dimension=1),
                        components.UnivariateGaussian(50, 5),
                    ]
                ),
                'num elements': self._compute_num_elements(30000),
            }
        self._components['filament-like structure'] = {
                'component': components.FilamentLikeStructure(
                    one_dimensional_components=[
                        components.UnivariateGaussian(50, 5),
                        components.UnivariateGaussian(50, 5),
                        components.UniformRandomNoise(0, 100, dimension=1),
                    ]
                ),
                'num elements': self._compute_num_elements(30000),
            }
