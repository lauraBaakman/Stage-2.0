import numpy as np

from datasets.simulated.simulateddataset import SimulatedDataSet
import datasets.simulated.components as components


class Ferdosi5(SimulatedDataSet):

    def __init__(self):
        super(Ferdosi5, self).__init__()

    def _init_components(self):
        self._components['wall-like structure 1'] = {
                'component': components.WallLikeStructure(
                    one_dimensional_components=[
                        components.UniformRandomNoise(0, 100, dimension=1),
                        components.UnivariateGaussian(10, 5),
                        components.UniformRandomNoise(0, 100, dimension=1),
                    ]
                ),
                'num elements': 20000,
            }
        self._components['wall-like structure 2'] = {
                'component': components.WallLikeStructure(
                    one_dimensional_components=[
                        components.UniformRandomNoise(0, 100, dimension=1),
                        components.UniformRandomNoise(0, 100, dimension=1),
                        components.UnivariateGaussian(50, 5),
                    ]
                ),
                'num elements': 20000,
            }
        self._components['wall-like structure 3'] = {
                'component': components.WallLikeStructure(
                    one_dimensional_components=[
                        components.UniformRandomNoise(0, 100, dimension=1),
                        components.UnivariateGaussian(50, 5),
                        components.UniformRandomNoise(0, 100, dimension=1)
                    ]
                ),
                'num elements': 20000,
            }
