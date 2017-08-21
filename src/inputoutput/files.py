import re
from enum import Enum

from unipath import Path


class _FileType(Enum):
    grid = 1
    dataset = 2
    gridResults = 3
    datasetResults = 4
    xisFile = 5


_dataset_name_regexp = '^(?P<semantic_name>[a-zA-Z]+_\d)_(?P<size>\d+)'
_estimator_regexp = '(?P<estimator>mbe|sambe|parzen)'
_sensitivity_regexp = '(?P<sensitivity>breiman|silverman)'
_grid_regexp = 'grid_(?P<grid_size>\d+)'


def parse_path(path):
    mapping = {
        _FileType.grid: _parse_grid_file_path,
        _FileType.dataset: _parse_dataset_file_path,
        _FileType.gridResults: _parse_grid_result_file_path,
        _FileType.datasetResults: _parse_dataset_result_file_path,
        _FileType.xisFile: _parse_xis_file_path
    }

    filetype = _determine_file_type(path)
    stem = _get_file_name_without_extension(path)
    return mapping.get(filetype)(stem)


def _determine_file_type(path):
    if is_dataset_file(path):
        return _FileType.dataset
    elif is_grid_file(path):
        return _FileType.grid
    elif is_dataset_result_file(path):
        return _FileType.datasetResults
    elif is_grid_result_file(path):
        return _FileType.gridResults
    elif is_xis_file(path):
        return _FileType.xisFile
    else:
        raise ValueError('{path} is not recognized as a specific type of path.'.format(path))


def _get_file_name_without_extension(path):
    return Path(path).stem


def _parse_helper(stem, regexp):
    def try_converting_value_to_int(key):
        if meta.get(key):
            meta[key] = int(meta[key])
        else:
            meta.pop(key, None)

    meta = re.match(regexp, stem).groupdict()
    try_converting_value_to_int(key='size')
    try_converting_value_to_int(key='grid_size')
    return meta


def _parse_grid_file_path(stem):
    reg_exp = '{dataset}_{grid}$'.format(
        dataset=_dataset_name_regexp,
        grid=_grid_regexp
    )
    return _parse_helper(stem, reg_exp)


def _parse_dataset_file_path(stem):
    reg_exp = '{dataset}$'.format(
        dataset=_dataset_name_regexp
    )
    return _parse_helper(stem, reg_exp)


def _parse_grid_result_file_path(stem):
    reg_exp = '{dataset}_{estimator}(_{sensitvity})?_{grid}$'.format(
                dataset=_dataset_name_regexp,
                estimator=_estimator_regexp,
                sensitvity=_sensitivity_regexp,
                grid=_grid_regexp
            )
    return _parse_helper(stem, reg_exp)


def _parse_dataset_result_file_path(stem):
    reg_exp = '{dataset}_{estimator}(_{sensitvity})?$'.format(
                dataset=_dataset_name_regexp,
                estimator=_estimator_regexp,
                sensitvity=_sensitivity_regexp
            )
    (dataset_name, size, estimator, _, sensitivity) = re.match(reg_exp, stem).groups()
    return _parse_helper(stem, reg_exp)


def _parse_xis_file_path(stem):
    reg_exp = '{dataset}_{estimator}(_{sensitivity})?(_{grid})?_xis$'.format(
                dataset=_dataset_name_regexp,
                estimator=_estimator_regexp,
                sensitivity=_sensitivity_regexp,
                grid=_grid_regexp
            )
    meta = _parse_helper(stem, reg_exp)
    meta['contains xis data'] = True
    return meta


def is_grid_file(path):
    stem = _get_file_name_without_extension(path)
    return bool(re.match(
           '{dataset}_{grid}$'.format(
                dataset=_dataset_name_regexp,
                grid=_grid_regexp),
           stem)
    )


def is_dataset_file(path):
    stem = _get_file_name_without_extension(path)
    return bool(re.match(
            '{dataset}$'.format(
                dataset=_dataset_name_regexp
            ),
            stem)
    )


def is_grid_result_file(path):
    stem = _get_file_name_without_extension(path)
    return bool(re.match(
            '{dataset}_{estimator}(_{sensitvity})?_{grid}$'.format(
                dataset=_dataset_name_regexp,
                estimator=_estimator_regexp,
                sensitvity=_sensitivity_regexp,
                grid=_grid_regexp
            ),
            stem)
    )


def is_dataset_result_file(path):
    stem = _get_file_name_without_extension(path)
    return bool(re.match(
            '{dataset}_{estimator}(_{sensitvitiy})?$'.format(
                dataset=_dataset_name_regexp,
                estimator=_estimator_regexp,
                sensitvitiy=_sensitivity_regexp
            ),
            stem)
    )


def is_xis_file(path):
    stem = _get_file_name_without_extension(path)
    return bool(re.match(
            '{dataset}_{estimator}(_{sensitivity})?(_{grid})?_xis$'.format(
                dataset=_dataset_name_regexp,
                estimator=_estimator_regexp,
                sensitivity=_sensitivity_regexp,
                grid=_grid_regexp
            ),
            stem)
    )


def is_xs_file(path):
    return is_grid_file(path) or is_dataset_file(path)


def is_results_file(path):
    return is_grid_result_file(path) or is_dataset_result_file(path)


def remove_skip_keys(keys, skip_keys):
    return keys - skip_keys


def _is_associated_file(dataset_meta, result_meta, skip_keys):
    skip_keys = skip_keys if skip_keys else list()
    dataset_keys = remove_skip_keys(dataset_meta.viewkeys(), skip_keys)
    result_keys = remove_skip_keys(result_meta.viewkeys(), skip_keys)

    if dataset_keys.difference(result_keys) is set():
        raise ValueError('The meta information of the data set and the result contain the same keys.')
    if 'grid_size' in dataset_keys ^ result_keys:
        return False
    shared_keys = dataset_keys & result_keys
    for key in shared_keys:
        if dataset_meta.get(key) != result_meta.get(key):
            return False
    return True


def is_associated_result_file(dataset_meta, result_meta, skip_keys=None):
    return (
        not result_meta.get('contains xis data', False) and
        _is_associated_file(dataset_meta, result_meta, skip_keys)
    )


def is_associated_xis_file(dataset_meta, result_meta, skip_keys=None):
    return (
            result_meta.get('contains xis data', False) and
            _is_associated_file(dataset_meta, result_meta, skip_keys)
    )
