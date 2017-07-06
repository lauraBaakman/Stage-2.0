import numpy as np

from datasets.simulated.simulateddataset import SimulatedDataSet
import datasets.simulated.components as components


class Baakman2(SimulatedDataSet):

    def __init__(self, scale=1.0):
        super(Baakman2, self).__init__(scale)

    def _init_components(self):
        self._components['trivariate gaussian 1'] = {
                'component': components.MultivariateGaussian(
                    mean=np.array([25, 25, 25]),
                    covariance_matrix=np.diag(
                        np.array([25, np.sqrt(5), np.sqrt(5)])
                    )
                ),
                'num elements': self._compute_num_elements(20000),
            }
        self._components['trivariate gaussian 2'] = {
                'component': components.MultivariateGaussian(
                    mean=np.array([65, 65, 65]),
                    covariance_matrix=np.diag(
                        np.array([np.sqrt(20), np.sqrt(20), 400])
                    )
                ),
                'num elements': self._compute_num_elements(20000),
        }
        self._components['uniform random noise'] = {
                'component': components.UniformRandomNoise(
                    minimum_value=-15,
                    maximum_value=150
                ),
                'num elements': self._compute_num_elements(20000),
        }
