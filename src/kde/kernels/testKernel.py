import  numpy as np

class TestKernel:

    def evaluate(self, x):
        return np.mean(x, axis=1)