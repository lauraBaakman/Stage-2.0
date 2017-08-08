from kde.datavalidation import EstimatorDataValidator


class EstimatorImplementation(object):

    def __init__(self, xi_s, x_s, dimension, kernel, general_bandwidth):
        super(EstimatorImplementation, self).__init__()
        self._dimension = dimension
        self._kernel = kernel
        # XI_S and X_S really need to be floats, otherwise the C module breaks
        self._xi_s = xi_s.astype(float, copy=False)
        self._general_bandwidth = general_bandwidth
        self._x_s = x_s.astype(float, copy=False)
        self._data_validator = EstimatorDataValidator(
            x_s=self._x_s, xi_s=self._xi_s
        ).validate()

    def estimate(self):
        raise NotImplementedError()

    @property
    def num_x_s(self):
        (n, _) = self._x_s.shape
        return n

    @property
    def num_xi_s(self):
        (n, _) = self._xi_s.shape
        return n

    @property
    def dimension(self):
        (_, dimension) = self._xi_s.shape
        return dimension

    @property
    def xis(self):
        return self._xi_s
