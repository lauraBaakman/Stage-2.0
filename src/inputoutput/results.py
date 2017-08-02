import numpy as np
import warnings


class Results:
    def __init__(self, densities=None, data_set=None, expected_size=None, num_used_patterns=None):
        if (expected_size is None) and (densities is None):
            raise TypeError("expected_size or densities need to be provided.'")
        if densities is not None:
            self._densities = densities
            self._num_used_patterns = num_used_patterns

            _DensitiesValidator(data_set=data_set, densities=densities).validate()
            self._incremental_result_adding_is_allowed = False
        if expected_size:
            self._densities = np.empty(expected_size)
            self._num_used_patterns = np.empty(expected_size)

            self._incremental_result_adding_is_allowed = True
            self._idx = 0

    @property
    def values(self):
        return self._densities

    @property
    def densities(self):
        return self._densities

    @property
    def num_used_patterns(self):
        return self._num_used_patterns

    @property
    def num_results(self):
        (num_results,) = self._densities.shape
        return num_results

    @property
    def has_num_used_patterns(self):
        return self._num_used_patterns is not None

    @property
    def is_incremental(self):
        return self._incremental_result_adding_is_allowed

    @property
    def xis(self):
        return self._xis

    @property
    def eigen_values(self):
        return self._eigen_values

    @property
    def eigen_vectors(self):
        return self._eigen_vectors

    def add_result(self, density, **kwargs):
        if not self.is_incremental:
            raise TypeError(
                'Adding results one by one is not allowed for this instance of the results object.'
            )
        self._add_density(density)
        self._add_num_used_patterns(
            kwargs.pop('num_used_patterns', np.NAN)
        )

        if len(kwargs.keys()) is not 0:
            raise KeyError(
                "The keys {} passed to 'add results' are not supported and thus ignored.".format(kwargs.keys())
            )

        self._idx += 1

    def _add_density(self, density):
        try:
            _DensitiesValidator.validate_density(density)
        except InvalidResultsException:
            warnings.warn('Adding the invalid density {} to the results.'.format(density))
        finally:
            self._densities[self._idx] = density

    def _add_num_used_patterns(self, num_used_patterns):
        self._num_used_patterns[self._idx] = num_used_patterns

    def to_file(self, out_file):
        _ResultsWriter(results=self, out_file=out_file).write()

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
        def eq_num_used_patterns(me, other):
            if (me is None) ^ (other is None):
                return False
            if (me is None) and (other is None):
                return True

            me_nans = np.isnan(me)
            other_nans = np.isnan(other)

            if np.any(me_nans != other_nans):
                return False

            return np.array_equiv(me[~me_nans], other[~other_nans])

        if isinstance(other, self.__class__):
            return (
                np.array_equiv(self._densities, other._densities) and
                eq_num_used_patterns(self._num_used_patterns, other._num_used_patterns)
            )
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)


class _ResultsReader(object):
    def __init__(self, in_file):
        self.in_file = in_file

    def read(self):
        densities, num_used_patterns = self._read_data()
        return Results(
            densities=densities,
            num_used_patterns=num_used_patterns
        )

    def _read_data(self):
        self.in_file.seek(0)
        try:
            data = np.genfromtxt(
                self.in_file,
                names=['densities', 'num_used_patterns']
            )
            num_used_patterns = data['num_used_patterns']
        except ValueError:
            self.in_file.seek(0)
            data = np.genfromtxt(self.in_file, names=['densities'])
            num_used_patterns = None
        finally:
            densities = data['densities']
        return densities, num_used_patterns


class _ResultsWriter(object):
    _densities_fmt = '%0.15f'
    _num_used_patterns_fmt = '%.0f'

    def __init__(self, results, out_file):
        self._out_file = out_file
        self._results = results

    @property
    def _values_as_matrix(self):
        def _add_densities(self):
            densities = self._results.densities
            format_string = self._densities_fmt
            return densities, format_string

        def _add_num_used_patterns(self, data, format_string):
            if self._results.has_num_used_patterns:
                data = np.vstack((data, self._results.num_used_patterns)).transpose()
                format_string = '{old} {new_column}'.format(
                    old=format_string,
                    new_column=self._num_used_patterns_fmt
                )
            return data, format_string

        data, format_string = _add_densities(self)
        data, format_string = _add_num_used_patterns(self, data, format_string)
        return data, format_string

    def write(self):
        data, format_string = self._values_as_matrix
        self._write_data(data, format_string)

    def _write_data(self, data, format_string):
        np.savetxt(self._out_file, data, fmt=format_string)


class _DensitiesValidator(object):
    def __init__(self, data_set, densities):
        self._data_set = data_set
        self._densities = densities

    def validate(self):
        if self._data_set:
            self._one_result_per_pattern()
        self._results_is_1D_array()
        self._results_are_densities()

    def _one_result_per_pattern(self):
        num_patterns = self._data_set.num_patterns
        (num_results,) = self._densities.shape
        if not num_results == num_patterns:
            raise InvalidResultsException(
                '''The number of results (should be equal to the number of patterns. The data set has {} patterns, '''
                '''and there are {} results.'''.format(num_patterns, num_results)
            )

    def _results_is_1D_array(self):
        if self._densities.ndim is not 1:
            raise InvalidResultsException(
                '''1D arrays are expected as results, the current results array '''
                '''has {} dimensions.'''.format(self._densities.ndim)
            )

    def _results_are_densities(self):
        def all_are_probability_densities(array):
            return np.all(array >= 0.0) and np.all(array <= 1.0)

        if not all_are_probability_densities(self._densities):
            warnings.warn("Not all values in the results are in the range [0, 1].")

    @staticmethod
    def validate_density(value):
        def is_density(value):
            return value >= 0.0 and value <= 1.0

        if not is_density(value):
            raise InvalidResultsException(
                message='{} is not a valid density'.format(value),
                actual=value,
                expected='Some value in the range [0 ,1].'
            )


class _XisValidator(object):
    def __init__(self, xis, eigen_vectors, eigen_values):
        self.xis = xis
        self.eigen_vectors = eigen_vectors
        self.eigen_values = eigen_values

    @property
    def xis_dimension(self):
        (_, dimension) = self.xis.shape
        return dimension

    @property
    def xis_count(self):
        (number_of_xis, _) = self.xis.shape
        return number_of_xis

    def validate(self):
        pass


class InvalidResultsException(Exception):
    def __init__(self, message, actual=None, expected=None, *args):
        self.message = message
        self.actual = actual
        self.expected = expected
        super(InvalidResultsException, self).__init__(message, *args)
