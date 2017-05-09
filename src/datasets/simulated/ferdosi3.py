import numpy as np

from datasets.simulated.simulateddataset import SimulatedDataSet
import datasets.simulated.components as components


class Ferdosi3(SimulatedDataSet):

    def __init__(self):
        super(Ferdosi3, self).__init__()

    def _init_components(self):
        self._components['trivariate gaussian 1'] = {
                'component': components.MultivariateGaussian(
                    mean=np.array([24, 10, 10]),
                    covariance_matrix=np.diag(np.array([2, 2, 2]))
                ),
                'num elements': 20000,
            }
        self._components['trivariate gaussian 2'] = {
                'component': components.MultivariateGaussian(
                    mean=np.array([33, 70, 40]),
                    covariance_matrix=np.diag(np.array([10, 10, 10]))
                ),
                'num elements': 20000,
        }
        self._components['trivariate gaussian 3'] = {
                'component': components.MultivariateGaussian(
                    mean=np.array([90, 20, 80]),
                    covariance_matrix=np.diag(np.array([1, 1, 1]))
                ),
                'num elements': 20000,
        }
        self._components['trivariate gaussian 4'] = {
                'component': components.MultivariateGaussian(
                    mean=np.array([60, 80, 23]),
                    covariance_matrix=np.diag(np.array([5, 5, 5]))
                ),
                'num elements': 20000,
        }
        self._components['uniform random noise'] = {
                'component': components.UniformRandomNoise(
                    minimum_value=0,
                    maximum_value=100
                ),
                'num elements': 40000,
        }
