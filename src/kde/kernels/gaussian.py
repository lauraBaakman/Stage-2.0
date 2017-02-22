from kde.kernels.kernel import KernelException, Kernel


class Gaussian(Kernel):
    def __init__(self, mean, covariance_matrix, implementation=None):
        implementation_class = implementation or _Gaussian_C
        self._implementation = implementation_class()
        self._mean = mean
        self._covariance_matrix = covariance_matrix

    def to_C_enum(self):
        return 3


class _Gaussian_C(Gaussian):
    def __init__(self):
        pass

    def evaluate(self, xs):
        raise NotImplementedError()

    def scaling_factor(self, general_bandwidth, eigen_values):
        raise NotImplementedError()


class _Gaussian_Python(Gaussian):
    def __init__(self, implementation=None):
        pass

    def evaluate(self, xs):
        raise NotImplementedError()

    def scaling_factor(self, general_bandwidth, eigen_values):
        raise NotImplementedError()
