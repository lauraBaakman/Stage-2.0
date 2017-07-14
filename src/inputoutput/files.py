import re
from enum import Enum

from unipath import Path


class _FileType(Enum):
    grid = 1
    dataset = 2
    gridResults = 3
    datasetResults = 4


_dataset_name_regexp = '^([a-zA-Z]+_\d)_(\d+)'
_estimator_regexp = '(mbe|sambe|parzen)'
_sensitivity_regexp = '(breiman|silverman)'


def parse_path(path):
    mapping = {
        _FileType.grid: _parse_grid_file_path,
        _FileType.dataset: _parse_dataset_file_path,
        _FileType.gridResults: _parse_grid_result_file_path,
        _FileType.datasetResults: _parse_dataset_result_file_path
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
    else:
        raise ValueError('{path} is not recognized as a specific type of path.'.format(path))


def _get_file_name_without_extension(path):
    return Path(path).stem


def _parse_grid_file_path(stem):
    reg_exp = '{dataset}_grid_(\d+)$'.format(
        dataset=_dataset_name_regexp
    )
    (dataset_name, size, grid_size) = re.match(reg_exp, stem).groups()
    return {
        'semantic name': dataset_name,
        'size': int(size),
        'grid size': int(grid_size)
    }


def _parse_dataset_file_path(stem):
    reg_exp = '{dataset}$'.format(
        dataset=_dataset_name_regexp
    )
    (dataset_name, size) = re.match(reg_exp, stem).groups()
    return {
        'semantic name': dataset_name,
        'size': int(size)
    }


def _parse_grid_result_file_path(stem):
    reg_exp = '{dataset}_{estimator}(_{sensitvity})?_grid_(\d+)$'.format(
                dataset=_dataset_name_regexp,
                estimator=_estimator_regexp,
                sensitvity=_sensitivity_regexp
            )
    (dataset_name, size, estimator, _, sensitivity, grid_size) = re.match(reg_exp, stem).groups()
    return {
        'semantic name': dataset_name,
        'size': int(size),
        'estimator': estimator,
        'sensitivity': sensitivity,
        'grid size': int(grid_size)
    }


def _parse_dataset_result_file_path(stem):
    reg_exp = '{dataset}_{estimator}(_{sensitvity})?$'.format(
                dataset=_dataset_name_regexp,
                estimator=_estimator_regexp,
                sensitvity=_sensitivity_regexp
            )
    (dataset_name, size, estimator, _, sensitivity) = re.match(reg_exp, stem).groups()
    return {
        'semantic name': dataset_name,
        'size': int(size),
        'estimator': estimator,
        'sensitivity': sensitivity,
    }


def is_grid_file(path):
    stem = _get_file_name_without_extension(path)
    return bool(re.match(
           '{dataset}_grid_\d+$'.format(
                dataset=_dataset_name_regexp),
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
            '{dataset}_{estimator}(_{sensitvity})?_grid_\d+$'.format(
                dataset=_dataset_name_regexp,
                estimator=_estimator_regexp,
                sensitvity=_sensitivity_regexp
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


def is_xs_file(path):
    return is_grid_file(path) or is_dataset_file(path)


def is_results_file(path):
    return is_grid_result_file(path) or is_dataset_result_file(path)


def remove_skip_keys(keys, skip_keys):
    return keys - skip_keys


def is_associated_result_file(dataset_meta, result_meta, skip_keys=None):
    skip_keys = skip_keys if skip_keys else list()
    dataset_keys = remove_skip_keys(dataset_meta.viewkeys(), skip_keys)
    result_keys = remove_skip_keys(result_meta.viewkeys(), skip_keys)

    if(len(dataset_keys) == len(result_keys)):
        raise ValueError('The meta information of the data set and the result contain the same keys.')
    if 'grid size' in dataset_keys ^ result_keys:
        return False
    shared_keys = dataset_keys & result_keys
    for key in shared_keys:
        if dataset_meta.get(key) != result_meta.get(key):
            return False
    return True
