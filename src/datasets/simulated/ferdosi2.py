import numpy as np

from datasets.simulated.simulateddataset import SimulatedDataSet
import datasets.simulated.components as components


class Ferdosi2(SimulatedDataSet):

    def __init__(self):
        super(Ferdosi2, self).__init__()

    def _init_components(self):
        self._components['trivariate gaussian 1'] = {
                'component': components.TrivariateGaussian(
                    mean=np.array([25, 25, 25]),
                    covariance_matrix=np.diag(np.array([5, 5, 5]))
                ),
                'num elements': 20000,
            }
        self._components['trivariate gaussian 2'] = {
                'component': components.TrivariateGaussian(
                    mean=np.array([65, 65, 65]),
                    covariance_matrix=np.diag(np.array([20, 20, 20]))
                ),
                'num elements': 20000,
        }
        self._components['uniform random noise'] = {
                'component': components.TrivariateUniformRandomNoise(
                    minimum_value=0,
                    maximum_value=100
                ),
                'num elements': 20000,
        }
