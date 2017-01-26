import numpy as np


class TestKernel:

    def evaluate(self, x):
        return np.abs(np.mean(x, axis=1))
