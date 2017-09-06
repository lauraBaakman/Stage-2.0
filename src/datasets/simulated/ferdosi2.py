import numpy as np

from datasets.simulated.simulateddataset import SimulatedDataSet
import datasets.simulated.components as components


class Ferdosi2(SimulatedDataSet):

    def __init__(self, scale=1.0):
        super(Ferdosi2, self).__init__(scale)

    def _init_components(self):
        self._components['trivariate gaussian 1'] = {
                'component': components.MultivariateGaussian(
                    mean=np.array([25, 25, 25]),
                    covariance_matrix=np.diag(np.array([5, 5, 5]))
                ),
                'num elements': self._compute_num_elements(20000),
            }
        self._components['trivariate gaussian 2'] = {
                'component': components.MultivariateGaussian(
                    mean=np.array([45, 45, 45]),
                    covariance_matrix=np.diag(np.array([11, 11, 11]))
                ),
                'num elements': self._compute_num_elements(20000),
        }
        self._components['uniform random noise'] = {
                'component': components.UniformRandomNoise(
                    minimum_value=0,
                    maximum_value=100
                ),
                'num elements': self._compute_num_elements(20000),
        }
