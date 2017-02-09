class DataSet(object):
    def __init__(self, patterns):
        self._patterns = patterns
        self._results = None
        _DataSetValidator(patterns=patterns).validate()

    @property
    def num_patterns(self):
        (num_patterns, _) = self._patterns.shape
        return num_patterns

    @property
    def dimension(self):
        (_, dimension) = self._patterns.shape
        return dimension

    @property
    def patterns(self):
        return self._patterns

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

    def _patterns_is_2D_array(self):
        if self._patterns.ndim is not 2:
            raise InvalidDataSetException(
                '''2D arrays are expected as input, the input array has {} dimensions.'''
                    .format(self._patterns.ndim)
            )


class InvalidDataSetException(Exception):
    def __init__(self, message, actual=None, expected=None, *args):
        self.message = message
        self.actual = actual
        self.expected = expected
        super(InvalidDataSetException, self).__init__(message, *args)
