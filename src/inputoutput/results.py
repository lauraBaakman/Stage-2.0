import numpy as np

class Results:
    def __init__(self, results_array, data_set=None):
        self._results_array = results_array
        if data_set:
            _ResultsValidator(data_set=data_set, results_array=results_array).validate()

    @property
    def num_results(self):
        (num_results,) = self._results_array.shape
        return num_results

    def to_file(self, out_file):
        _ResultsWriter(results=self._results_array, out_file=out_file).write()


class _ResultsWriter(object):
    def __init__(self, results, out_file):
        self._out_file = out_file
        self._results = results

    def write(self):
        self._write_densities()

    def _write_densities(self):
        np.savetxt(self._out_file, self._results, fmt='%.18f')


class _ResultsValidator(object):
    def __init__(self, data_set, results_array):
        self._data_set = data_set
        self._results_array = results_array

    def validate(self):
        self._one_result_per_pattern()
        self._results_is_1D_array()

    def _one_result_per_pattern(self):
        num_patterns = self._data_set.num_patterns
        (num_results,) = self._results_array.shape
        if not num_results == num_patterns:
            raise InvalidResultsException(
                '''The number of results (should be equal to the number of patterns. The data set has {} patterns, '''
                '''and there are {} results.'''
                    .format(num_patterns, num_results)
            )

    def _results_is_1D_array(self):
        if self._results_array.ndim is not 1:
            raise InvalidResultsException(
                '''1D arrays are expected as results, the current results array has {} dimensions.'''
                    .format(self._results_array.ndim)
            )


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