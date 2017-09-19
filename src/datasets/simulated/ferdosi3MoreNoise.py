import numpy as np

from datasets.simulated.simulateddataset import SimulatedDataSet
import datasets.simulated.components as components


class Ferdosi3MoreNoise(SimulatedDataSet):

    def __init__(self, scale=1.0):
        super(Ferdosi3MoreNoise, self).__init__(scale)

    def _init_components(self):
        self._components['trivariate gaussian 1'] = {
                'component': components.MultivariateGaussian(
                    mean=np.array([24, 10, 10]),
                    covariance_matrix=np.diag(np.array([2, 2, 2]))
                ),
                'num elements': self._compute_num_elements(20000),
            }
        self._components['trivariate gaussian 2'] = {
                'component': components.MultivariateGaussian(
                    mean=np.array([33, 70, 40]),
                    covariance_matrix=np.diag(np.array([10, 10, 10]))
                ),
                'num elements': self._compute_num_elements(20000),
        }
        self._components['trivariate gaussian 3'] = {
                'component': components.MultivariateGaussian(
                    mean=np.array([90, 20, 80]),
                    covariance_matrix=np.diag(np.array([1, 1, 1]))
                ),
                'num elements': self._compute_num_elements(20000),
        }
        self._components['trivariate gaussian 4'] = {
                'component': components.MultivariateGaussian(
                    mean=np.array([60, 80, 23]),
                    covariance_matrix=np.diag(np.array([5, 5, 5]))
                ),
                'num elements': self._compute_num_elements(20000),
        }
        self._components['uniform random noise'] = {
                'component': components.UniformRandomNoise(
                    minimum_value=-20,
                    maximum_value=120
                ),
                'num elements': self._compute_num_elements(40000 * 2.744),
        }
