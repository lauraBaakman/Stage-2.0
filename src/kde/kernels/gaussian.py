from kde.kernels.kernel import Kernel


class Gaussian(object):

    def __new__(cls, mean, covariance_matrix, implementation=None):
        implementation_class = implementation or _Gaussian_C
        return implementation_class(mean=mean, covariance_matrix=covariance_matrix)


class _Gaussian(Kernel):

    def __init__(self, mean, covariance_matrix):
        self._mean = mean
        self._covariance_matrix = covariance_matrix

    def to_C_enum(self):
        return 3


class _Gaussian_C(_Gaussian):
    def __init__(self, *args, **kwargs):
        super(_Gaussian_C, self).__init__(*args, **kwargs)

    def evaluate(self, xs):
        raise NotImplementedError()

    def scaling_factor(self, general_bandwidth, eigen_values):
        raise NotImplementedError()


class _Gaussian_Python(Kernel):
    def __init__(self, *args, **kwargs):
        super(_Gaussian_Python, self).__init__(*args, **kwargs)

    def evaluate(self, xs):
        raise NotImplementedError()

    def scaling_factor(self, general_bandwidth, eigen_values):
        raise NotImplementedError()
