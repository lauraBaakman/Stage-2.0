class Estimator(object):

    def __init__(self):
        super(Estimator, self).__init__()

    def _validate_data(self, x_s, xi_s):
        self._validate_data_set(x_s)
        self._validate_data_set(xi_s)
        self._have_same_dimension(x_s, xi_s)

    def _have_same_dimension(self, x_s, xi_s):
        (_, dim_x_s) = x_s.shape
        (_, dim_xi_s) = xi_s.shape
        if dim_x_s is not dim_xi_s:
            raise InvalidEstimatorArguments('''
                The estimator expects the data points and the patterns to have the same dimensionality, but the data points have dimension
                {} and the patterns have dimension {}.'''.format(dim_xi_s, dim_x_s)
            )

    def _validate_data_set(self, data):
        self._array_has_correct_shape(data)

    def _array_has_correct_shape(self, x_s):
        if x_s.ndim is not 2:
            raise InvalidEstimatorArguments('''
                The estimator expects 2D arrays as input, the input array has {} dimensions.
            '''.format(x_s.ndim))


class InvalidEstimatorArguments(Exception):
    pass
