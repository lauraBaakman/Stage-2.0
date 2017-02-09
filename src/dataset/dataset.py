class DataSet(object):
    def __init__(self, patterns):
        self._patterns = patterns
        self._results = None
        _DataSetValidator(patterns=patterns).validate()

    @property
    def num_patterns(self):
        (num_patterns, _) = self.patterns.shape
        return num_patterns

    @property
    def dimension(self):
        (_, dimension) = self.patterns.shape
        return dimension

    @property
    def patterns(self):
        return self.patterns

    @classmethod
    def from_file(cls, in_file):
        return _DataSetReader(in_file).read()


class _DataSetReader(object):
    def __init__(self, in_file):
        self.in_file = in_file

    def read(self):
        raise NotImplementedError()


class _DataSetValidator(object):
    def __init__(self, patterns):
        self._patterns = patterns

    def validate(self):
        self._patterns_is_2D_array()
        self._each_pattern_has_same_dimension()

    def _patterns_is_2D_array(self):
        if self._patterns.ndim is not 2:
            raise InvalidDataSetException(
                '''2D arrays are expected as input, the input array has {} dimensions.'''
                    .format(self._data_set.patterns.ndim)
            )

    def _each_pattern_has_same_dimension(self):
        first_element_size = self._patterns[0].shape[0]
        for element in self._patterns[1:]:
            current_element_size = element.shape[0]
            if element is not first_element_size:
                raise InvalidDataSetException(
                    '''Each element of the data set is expected to have dimension {}, an element with dimension {} '''
                    '''was encountered.'''
                        .format(first_element_size, current_element_size))


class InvalidDataSetException(Exception):
    def __init__(self, message, actual=None, expected=None, *args):
        self.message = message
        self.actual = actual
        self.expected = expected
        super(InvalidDataSetException, self).__init__(message, *args)
