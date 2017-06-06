import numpy as np

from datasets.simulated.simulateddataset import SimulatedDataSet
import datasets.simulated.components as components


class Baakman5(SimulatedDataSet):

    def __init__(self):
        super(Baakman5, self).__init__()

    def _init_components(self):
        self._components['trivariate gaussian 1'] = {
                'component': components.MultivariateGaussian(
                    mean=np.array([50, 50, 50]),
                    covariance_matrix=np.diag(np.array(
                        [9, 3, 1])
                    )
                ),
                'num elements': 40000,
            }
        self._components['uniform random noise'] = {
                'component': components.UniformRandomNoise(
                    minimum_value=0,
                    maximum_value=100
                ),
                'num elements': 20000,
        }