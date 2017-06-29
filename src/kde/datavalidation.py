class InvalidEstimatorArguments(Exception):
    def __init__(self, message, actual=None, expected=None, *args):
        self.message = message
        self.actual = actual
        self.expected = expected
        super(InvalidEstimatorArguments, self).__init__(message, *args)


class EstimatorDataValidator(object):

    def __init__(self, x_s, xi_s):
        self._x_s = x_s
        self._xi_s = xi_s

    def validate(self):
        self._validate_data_set(self._x_s, 'x_s')
        self._validate_data_set(self._xi_s, 'xi_s')
        self._do_elements_have_same_dimension(self._x_s, self._xi_s)

    def _validate_data_set(self, data_set, name='<variable name>'):
        self._array_has_two_dimensions(data_set, name)
        self._array_has_correct_flags(data_set, name)

    @staticmethod
    def _array_has_two_dimensions(array, name='<variable name>'):
        if array.ndim is not 2:
            raise InvalidEstimatorArguments(
                '''The estimator expects 2D arrays as input, the input array {name} '''
                '''has {dimension} dimensions.'''.format(
                    name=name,
                    dimension=array.ndim
                )
            )

    @classmethod
    def _array_has_correct_flags(cls, array, name='<variable name>'):
        cls._array_has_C_order(array, name)
        cls._array_owns_data(array, name)

    @staticmethod
    def _array_has_C_order(array, name='<variable name>'):
        if not array.flags['C_CONTIGUOUS']:
            raise InvalidEstimatorArguments(
                'The input array {} is not C-contiguous'.format(name)
            )

    @staticmethod
    def _array_owns_data(array, name='<variable name>'):
        if not array.flags['OWNDATA']:
            raise InvalidEstimatorArguments(
                'The input array {} does not own its data.'.format(name)
            )

    @classmethod
    def _do_elements_have_same_dimension(cls, *arrays):
        try:
            cls._do_arrays_have_same_size_in_dimension(1, *arrays)
        except InvalidEstimatorArguments as e:
            raise InvalidEstimatorArguments(
                '''The estimator expects the elements to have dimension {expected}, but '''
                '''some array has elements with dimension {actual}.'''.format(
                    expected=e.expected,
                    actual=e.actual
                )
            )

    @staticmethod
    def _do_arrays_have_same_size_in_dimension(dimension, *arrays):
        first_element_size = arrays[0].shape[dimension]
        for array in arrays[1:]:
            current_array_size = array.shape[dimension]
            if current_array_size is not first_element_size:
                raise InvalidEstimatorArguments(None, actual=current_array_size, expected=first_element_size)


class ParzenDataValidator(EstimatorDataValidator):

    def __init__(self, x_s, xi_s):
        super(ParzenDataValidator, self).__init__(x_s, xi_s)


class MBEDataValidator(EstimatorDataValidator):

    def __init__(self, x_s, xi_s, local_bandwidths):
        super(MBEDataValidator, self).__init__(x_s, xi_s)
        self._local_bandwidths = local_bandwidths

    def validate(self):
        super(MBEDataValidator, self).validate()
        self._do_arrays_have_the_same_length(self._xi_s, self._local_bandwidths)

    @classmethod
    def _do_arrays_have_the_same_length(cls, *arrays):
        try:
            cls._do_arrays_have_same_size_in_dimension(0, *arrays)
        except InvalidEstimatorArguments as e:
            raise InvalidEstimatorArguments(
                '''The estimator expects the elements to have the same length, namely '''
                '''{expected}, but some array has length {actual}.'''.format(
                    expected=e.expected,
                    actual=e.actual
                )
            )
