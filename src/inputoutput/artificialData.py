import io
import sys
import numpy as np


_header_size = 2


def read(input_file):
    """
    Read artificial data from the file *input_file*. Expected structure of the file:
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
    pattern_count = _read_pattern_count(input_file)
    patterns = _read_patterns(input_file, pattern_count)
    labels = _read_labels(input_file, pattern_count)
    return patterns, labels


def _read_pattern_count(input_file):
    with open(input_file) as file_handle:
        line = file_handle.readline()
        return int(line.split()[0])


def _abstract_read(input_file, skip_header, pattern_count):
    with io.open(input_file, 'rb') as file_handle:
        try:
            return np.genfromtxt(file_handle, skip_header=skip_header, max_rows=pattern_count, invalid_raise=True)
        except ValueError as e:
            sys.stderr.write(
                "Got an error while reading the file '{file}':\n"
                "{error}\n".format(
                    file=input_file, error=e))


def _read_patterns(input_file, pattern_count):
    return _abstract_read(input_file,
                          skip_header=_header_size,
                          pattern_count=pattern_count)


def _read_labels(input_file, pattern_count):
    return _abstract_read(input_file,
                          skip_header=_header_size + pattern_count,
                          pattern_count=pattern_count)

if __name__ == '__main__':
    input_file = '/Users/laura/Repositories/stage-2.0/data/artificial/test.txt'
    patterns, labels = read(input_file)
