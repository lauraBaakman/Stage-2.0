import numpy as np
import warnings
import re

_delimiter = ' '


class Results:
    def __init__(self, densities=None, data_set=None, expected_size=None, num_used_patterns=None,
                 xis=None, eigen_values=None, eigen_vectors=None, scaling_factors=None):
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

        self._xis = xis
        self._eigen_values = eigen_values
        self._eigen_vectors = eigen_vectors
        self._scaling_factors = scaling_factors

        _XisValidator(
            xis=self._xis,
            eigen_values=self._eigen_values,
            eigen_vectors=self._eigen_vectors,
            scaling_factors=self._scaling_factors
        ).validate()

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

    @xis.setter
    def xis(self, xis):
        self._xis = xis

    @property
    def eigen_values(self):
        return self._eigen_values

    @property
    def eigen_vectors(self):
        return self._eigen_vectors

    @property
    def scaling_factors(self):
        return self._scaling_factors

    @property
    def dimension(self):
        (_, dimension) = self._xis.shape
        return dimension

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

    def to_file(self, x_out_file, xi_out_file=None):
        if xi_out_file and self.dimension != 3:
            warnings.warn(
                'Writing xi data for non 3D data is not supported, skipping the writing of xi data.'
            )
            xi_out_file = None
        _ResultsWriter(
            results=self,
            x_out_file=x_out_file,
            xi_out_file=xi_out_file
        ).write()

    @classmethod
    def from_file(cls, x_file, xi_file=None):
        try:
            results = _ResultsReader(x_file, xi_file).read()
        except AttributeError:
            with open(x_file, mode='rb') as x_handle:
                if xi_file is not None:
                    with open(xi_file, mode='rb') as xi_handle:
                        results = _ResultsReader(x_handle, xi_handle).read()
                else:
                    results = _ResultsReader(x_handle).read()
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

        def eq_numpy_array_that_can_be_none(me, other):
            if (me is None) ^ (other is None):
                return False
            if (me is None) and (other is None):
                return True
            if me.shape != other.shape:
                return False
            return np.allclose(me, other)

        if isinstance(other, self.__class__):
            if self._densities.shape != other._densities.shape:
                return False

            return (
                np.allclose(self._densities, other._densities) and
                eq_num_used_patterns(self._num_used_patterns, other._num_used_patterns) and
                eq_numpy_array_that_can_be_none(self._xis, other._xis) and
                eq_numpy_array_that_can_be_none(self._eigen_values, other._eigen_values) and
                eq_numpy_array_that_can_be_none(self._eigen_vectors, other._eigen_vectors) and
                eq_numpy_array_that_can_be_none(self._scaling_factors, other._scaling_factors)
            )
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)


class _ResultsReader(object):
    def __init__(self, x_in_file, xi_in_file=None):
        self._x_in_file = x_in_file
        self._xi_in_file = xi_in_file

    def read(self):
        data = self._read_x_data()
        data.update(
            dict()
            if self._xi_in_file is None
            else self._read_xi_data()
        )
        return Results(**data)

    def _read_x_data(self):
        self._x_in_file.seek(0)
        try:
            data = np.genfromtxt(
                self._x_in_file,
                names=['densities', 'num_used_patterns']
            )
            num_used_patterns = data['num_used_patterns']
        except ValueError:
            self._x_in_file.seek(0)
            data = np.genfromtxt(self._x_in_file, names=['densities'])
            num_used_patterns = None
        finally:
            densities = data['densities']
        return {'densities': densities,
                'num_used_patterns': num_used_patterns}

    def _read_xi_data(self):
        def contains_regular_expression(data, regex):
            return any([regex.findall(column_name) for column_name in data.dtype.names])

        def contains_eigen_values(data):
            regex = re.compile('eigen_value_\d')
            return contains_regular_expression(data, regex)

        def get_eigen_values(data):
            if contains_eigen_values(data):
                return np.array((data['eigen_value_1'], data['eigen_value_2'], data['eigen_value_3'])).T
            return None

        def contains_eigen_vectors(data):
            regex = re.compile('eigen_vector_\d_[xyz]')
            return contains_regular_expression(data, regex)

        def get_eigen_vectors(data):
            if contains_eigen_vectors(data):
                eigen_vectors = np.array((
                    data['eigen_vector_1_x'], data['eigen_vector_2_x'], data['eigen_vector_3_x'],
                    data['eigen_vector_1_y'], data['eigen_vector_2_y'], data['eigen_vector_3_y'],
                    data['eigen_vector_1_z'], data['eigen_vector_2_z'], data['eigen_vector_3_z'],
                )).T
                (num_xis, ) = data.shape
                return np.reshape(eigen_vectors, (num_xis, 3, 3))
            return None

        def contains_scaling_factors(data):
            regex = re.compile('scaling_factor')
            return contains_regular_expression(data, regex)

        def get_scaling_factors(data):
            if contains_scaling_factors(data):
                return np.array((data['scaling_factor'])).T
            return None

        data = np.genfromtxt(
                             self._xi_in_file,
                             delimiter=_delimiter,
                             names=True, autostrip=True, case_sensitive=True,
                             unpack=True)
        return {
            'xis': np.array((data['xi_x'], data['xi_y'], data['xi_z'])).T,
            'eigen_values': get_eigen_values(data),
            'eigen_vectors': get_eigen_vectors(data),
            'scaling_factors': get_scaling_factors(data)
        }


class _ResultsWriter(object):
    _float_fmt = '%0.15f'
    _num_used_patterns_fmt = '%.0f'

    def __init__(self, results, x_out_file, xi_out_file=None):
        self._x_out_file = x_out_file
        self._xi_out_file = xi_out_file
        self._results = results

    def write(self):
        self._write_x_data()
        if self._xi_out_file is not None:
            self._write_xi_data()

    def _write_x_data(self):
        data, format_string = self._x_values_to_matrix()
        np.savetxt(self._x_out_file, data, fmt=format_string)

    def _x_values_to_matrix(self):
        def _add_densities(self):
            densities = self._results.densities
            format_string = self._float_fmt
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

    def _write_xi_data(self):
        data, header = self._xi_values_to_matrix()
        np.savetxt(self._xi_out_file, data,
                   fmt=self._float_fmt, header=header, delimiter=_delimiter)

    def _xi_values_to_matrix(self):
        def add_eigen_values(eigen_values):
            if eigen_values is not None:
                column_headers.extend(['eigen_value_1', 'eigen_value_2', 'eigen_value_3'])
                return np.hstack((matrix, eigen_values)), column_headers
            return matrix, column_headers

        def add_eigen_vectors(eigen_vectors):
            if eigen_vectors is not None:
                column_headers.extend(['eigen_vector_1_x', 'eigen_vector_2_x', 'eigen_vector_3_x'])
                column_headers.extend(['eigen_vector_1_y', 'eigen_vector_2_y', 'eigen_vector_3_y'])
                column_headers.extend(['eigen_vector_1_z', 'eigen_vector_2_z', 'eigen_vector_3_z'])

                (num_xis, num_eigen_vectos, dimension) = eigen_vectors.shape
                return np.hstack((
                    matrix,
                    np.reshape(eigen_vectors, (num_xis, num_eigen_vectos * dimension))
                )), column_headers
            return matrix, column_headers

        def add_scaling_factors(scaling_factors):
            if scaling_factors is not None:
                column_headers.extend(['scaling_factor'])
                num_xis, = scaling_factors.shape
                return np.hstack((
                    matrix,
                    np.reshape(scaling_factors, (num_xis, 1))
                )), column_headers
            return matrix, column_headers

        matrix = np.array(
            self._results.xis,
        )
        column_headers = ['xi_x', 'xi_y', 'xi_z']

        matrix, column_headers = add_eigen_values(self._results.eigen_values)
        matrix, column_header = add_eigen_vectors(self._results.eigen_vectors)
        matrix, column_header = add_scaling_factors(self._results.scaling_factors)
        return matrix, _delimiter.join(column_headers)


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
    def __init__(self, xis, eigen_vectors=None, eigen_values=None, scaling_factors=None):
        self.xis = xis
        self.eigen_vectors = eigen_vectors
        self.eigen_values = eigen_values
        self.scaling_factors = scaling_factors

    @property
    def xis_dimension(self):
        (_, dimension) = self.xis.shape
        return dimension

    @property
    def xis_count(self):
        (number_of_xis, _) = self.xis.shape
        return number_of_xis

    @property
    def has_xis(self):
        return self.xis is not None

    def validate(self):
        self._validate_eigen_vectors()
        self._validate_eigen_values()
        self._validate_scaling_factors()

    def _validate_eigen_vectors(self):
        if self.eigen_vectors is not None:
            self._validate_eigen_vectors_ndim()
            if self.has_xis:
                self._validate_total_number_of_eigen_vectors()
                self._validate_number_of_eigen_vectors()
                self._validate_eigen_vector_dimension()

    def _validate_eigen_vectors_ndim(self):
        eigen_vector_ndim = self.eigen_vectors.ndim
        if not (eigen_vector_ndim == 3):
            raise InvalidResultsException(
                'The eigenvectors should be stored in a 3D array.'
            )

    def _validate_total_number_of_eigen_vectors(self):
        (eigen_vector_count, _, _) = self.eigen_vectors.shape
        if not (eigen_vector_count == self.xis_count):
            raise InvalidResultsException(
                '{num_xis} xis and {num_eigen_vectors} sets of eigen vectors is not a valid combination'.format(
                    num_xis=self.xis_count,
                    num_eigen_vectors=eigen_vector_count
                )
            )

    def _validate_number_of_eigen_vectors(self):
        (_, eigen_vector_count, _) = self.eigen_vectors.shape
        if not (eigen_vector_count == self.xis_dimension):
            raise InvalidResultsException(
                'Xis of dimension {dim} and {num_eigen_vectors} per pattern is not a valid combination'.format(
                    dim=self.xis_dimension,
                    num_eigen_vectors=eigen_vector_count
                )
            )

    def _validate_eigen_vector_dimension(self):
        (_, _, eigen_vector_dimension) = self.eigen_vectors.shape
        if not (eigen_vector_dimension == self.xis_dimension):
            raise InvalidResultsException(
                'Xis of dimension {} and eigen vectors of dimension {} is not a valid combination'.format(
                    self.xis_dimension, eigen_vector_dimension
                )
            )

    def _validate_eigen_values(self):
        if self.eigen_values is not None:
            self._validate_eigen_values_ndim()
            if self.has_xis:
                self._validate_total_number_of_eigen_values()
                self._validate_number_of_eigen_values()

    def _validate_eigen_values_ndim(self):
        ndim = self.eigen_values.ndim
        if not (ndim == 2):
            raise InvalidResultsException(
                'The eigenvalues should be stored in a 2D array.'
            )

    def _validate_total_number_of_eigen_values(self):
        (eigen_value_count, _) = self.eigen_values.shape
        if not (eigen_value_count == self.xis_count):
            raise InvalidResultsException(
                '{num_xis} xis and {num_eigen_vectors} sets of eigen values is not a valid combination'.format(
                    num_xis=self.xis_count,
                    num_eigen_vectors=eigen_value_count
                )
            )

    def _validate_number_of_eigen_values(self):
        (_, eigen_value_count) = self.eigen_values.shape
        if not (eigen_value_count == self.xis_dimension):
            raise InvalidResultsException(
                'Xis of dimension {dim} and {num_eigen_values} per pattern is not a valid combination'.format(
                    dim=self.xis_dimension,
                    num_eigen_values=eigen_value_count
                )
            )

    def _validate_scaling_factors(self):
        if self.scaling_factors is not None:
            self._validate_scaling_factors_dimension()
            if self.has_xis:
                self._validate_total_number_of_scaling_factors()

    def _validate_total_number_of_scaling_factors(self):
        (scaling_factor_count,) = self.scaling_factors.shape
        if not (scaling_factor_count == self.xis_count):
            raise InvalidResultsException(
                '{num_xis} xis and {num_scaling_factors} scaling factors is not a valid combination'.format(
                    num_xis=self.xis_count,
                    num_scaling_factors=scaling_factor_count
                )
            )

    def _validate_scaling_factors_dimension(self):
        ndim = self.scaling_factors.ndim
        if not (ndim == 1):
            raise InvalidResultsException(
                'The scaling factors should be stored in a 1D array.'
            )


class InvalidResultsException(Exception):
    def __init__(self, message, actual=None, expected=None, *args):
        self.message = message
        self.actual = actual
        self.expected = expected
        super(InvalidResultsException, self).__init__(message, *args)
