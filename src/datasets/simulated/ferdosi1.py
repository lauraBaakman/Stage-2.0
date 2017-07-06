import numpy as np

from datasets.simulated.simulateddataset import SimulatedDataSet
import datasets.simulated.components as components


class Ferdosi1(SimulatedDataSet):

    def __init__(self, scale=1.0):
        super(Ferdosi1, self).__init__(scale)

    def _init_components(self):
        self._components['trivariate gaussian 1'] = {
                'component': components.MultivariateGaussian(
                    mean=np.array([50, 50, 50]),
                    covariance_matrix=np.diag(np.array([30, 30, 30]))
                ),
                'num elements': self._compute_num_elements(40000),
            }
        self._components['uniform random noise'] = {
                'component': components.UniformRandomNoise(
                    minimum_value=0,
                    maximum_value=100
                ),
                'num elements': self._compute_num_elements(20000),
        }
