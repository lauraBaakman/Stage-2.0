import numpy as np
import warnings


class Results:
    def __init__(self, results_array, data_set=None):
        self._results_array = results_array
        _ResultsValidator(data_set=data_set, results_array=results_array).validate()

    @property
    def values(self):
        return self._results_array

    @property
    def num_results(self):
        (num_results,) = self._results_array.shape
        return num_results

    def to_file(self, out_file):
        _ResultsWriter(results=self._results_array, out_file=out_file).write()

    @classmethod
    def from_file(cls, in_file):
        try:
            results = _ResultsReader(in_file).read()
        except AttributeError:
            with open(in_file, mode='rb') as input_file_handle:
                results = _ResultsReader(input_file_handle).read()
        return results

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                np.array_equiv(self._results_array, other._results_array)
            )
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented


class _ResultsReader(object):
    def __init__(self, in_file):
        self.in_file = in_file

    def read(self):
        densities = self._read_densities()
        return Results(results_array=densities)

    def _read_densities(self):
        self.in_file.seek(0)
        return np.genfromtxt(
            self.in_file,
            skip_header=0, invalid_raise=True
        )


class _ResultsWriter(object):
    def __init__(self, results, out_file):
        self._out_file = out_file
        self._results = results

    def write(self):
        self._write_densities()

    def _write_densities(self):
        np.savetxt(self._out_file, self._results, fmt='%.15f')


class _ResultsValidator(object):
    def __init__(self, data_set, results_array):
        self._data_set = data_set
        self._results_array = results_array

    def validate(self):
        if self._data_set:
            self._one_result_per_pattern()
        self._results_is_1D_array()
        self._results_are_densities()

    def _one_result_per_pattern(self):
        num_patterns = self._data_set.num_patterns
        (num_results,) = self._results_array.shape
        if not num_results == num_patterns:
            raise InvalidResultsException(
                '''The number of results (should be equal to the number of patterns. The data set has {} patterns, '''
                '''and there are {} results.'''.format(num_patterns, num_results)
            )

    def _results_is_1D_array(self):
        if self._results_array.ndim is not 1:
            raise InvalidResultsException(
                '''1D arrays are expected as results, the current results array '''
                '''has {} dimensions.'''.format(self._results_array.ndim)
            )

    def _results_are_densities(self):
        def all_are_probability_densities(array):
            return np.all(array >= 0.0) and np.all(array <= 1.0)

        if not all_are_probability_densities(self._results_array):
            warnings.warn("Not all values in the results are in the range [0, 1].")


class InvalidResultsException(Exception):
    def __init__(self, message, actual=None, expected=None, *args):
        self.message = message
        self.actual = actual
        self.expected = expected
        super(InvalidResultsException, self).__init__(message, *args)


if __name__ == '__main__':
    output_file = '/Users/laura/Desktop/temp.txt'
    results = Results(results_array=np.array([1.0, 2.0, 3.0, 4.0]))

    # Option 1
    # with open(output_file) as out_file_object:
    #     results.to_file(output_file)

    # Option 2
    results.to_file(output_file)
