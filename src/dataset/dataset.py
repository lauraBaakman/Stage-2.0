import numpy as np


class DataSet(object):
    def __init__(self, patterns):
        self._patterns = patterns
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
        """
        Read artificial data from the file *in_file*. Expected structure of the file:
        number_of_patterns pattern_dimensionality
        pattern 1
        pattern 2
        ....
        pattern number_of_patterns
        label 1
        label 2
        ....
        label number_of_patterns

        :param input_file: The file to read the data from.
        :return: patterns, labels
        """
        return _DataSetReader(in_file).read()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return np.array_equiv(self.patterns, other.patterns)
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented


class _DataSetReader(object):
    _header_size = 2

    def __init__(self, in_file):
        self.in_file = in_file
        self._num_patterns = None

    def read(self):
        self._num_patterns = self._read_pattern_count()
        patterns = self._read_patterns()
        return DataSet(patterns=patterns)

    def _read_pattern_count(self):
        line = self.in_file.readline()
        return int(line.split()[0])

    def _read_patterns(self):
        return self._abstract_read(num_rows_to_skip=self._header_size)

    def _read_labels(self):
        return self._abstract_read(num_rows_to_skip=self._header_size + self._num_patterns)

    def _abstract_read(self, num_rows_to_skip):
        self.in_file.seek(0)
        return np.genfromtxt(
            self.in_file,
            skip_header=num_rows_to_skip, max_rows=self._num_patterns, invalid_raise=True
        )


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


if __name__ == '__main__':
    input_file = '/Users/laura/Repositories/stage-2.0/data/artificial/test.txt'
    with open(input_file, mode='rb') as in_file:
        data_set = DataSet.from_file(in_file)