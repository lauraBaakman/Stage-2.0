from unipath import Path

import files as filenames
import numpy as np


def _build_result_file_stem(data_set_file_path, estimator, *args):
    args_string = '_'.join(args)
    stem = '{data_set}_{estimator}{seperator}{args_string}'.format(
        data_set=Path(data_set_file_path).stem,
        estimator=estimator,
        seperator='_' if args_string else '',
        args_string=args_string
    )
    return stem


def build_x_result_data_path(results_directory, data_set_file_path, estimator, *args):
    stem = _build_result_file_stem(data_set_file_path, estimator, *args)
    out_file_name = '{stem}.txt'.format(stem=stem)
    return results_directory.child(out_file_name)


def build_xi_result_data_path(results_directory, data_set_file_path, estimator, *args):
    stem = _build_result_file_stem(data_set_file_path, estimator, *args)
    out_file_name = '{stem}_xis.txt'.format(stem=stem)
    return results_directory.child(out_file_name)


def build_xs_path(directory, data_set_file_path, *args):
    args_string = '_'.join(args)
    out_file_name = '{data_set}{seperator}{args_string}.txt'.format(
        data_set=Path(data_set_file_path).stem,
        seperator='_' if args_string else '',
        args_string=args_string
    )
    return directory.child(out_file_name)


def partial_path(path):
    return Path(path.components()[-3:])


def get_data_set_files(input_path, ask_for_confirmation=False, show_files=True):
    files = list(input_path.walk(filter=lambda x: filenames.is_xs_file(x)))
    if ask_for_confirmation:
        files = _confirm_files(files)
    if show_files:
        _show_files_to_user(files)
    return files


def get_result_files(input_path, ask_for_confirmation=False, show_files=True):
    files = list(input_path.walk(filter=lambda x: filenames.is_results_file(x)))
    if ask_for_confirmation:
        files = _confirm_files(files)
    if show_files:
        _show_files_to_user(files)
    return files


def _confirm_files(potential_files):
    def add_data_set(data_set):
        files.append(data_set)

    def skip_data_set(*args, **kwargs):
        pass

    responses = {
        'y': add_data_set,
        'n': skip_data_set
    }

    files = list()

    for data_set in potential_files:
        response = raw_input(
            'Include the data set in the file ..{}? [y/N] '.format(
                partial_path(data_set))
        ).lower()
        responses.get(response, skip_data_set)(data_set)
    print('\n')
    return files


def _show_files_to_user(files):
    if not files:
        print("No files selected")
    print("Running the experiment on:\n{data_sets}\n".
          format(
                 data_sets='\n'.join([
                                     "\t{}".format(partial_path(file))
                                     for file
                                     in files])
                 )
          )


def sub_sample_grid(data, space, dimension=3):
    def subsample_along_axis(data, space, axis):
        unique = np.unique(data[:, axis])
        values_that_stay = unique[::space + 1]
        return data[np.in1d(data[:, axis], values_that_stay)]

    for axis in range(dimension):
        data = subsample_along_axis(data, space, axis=axis)

    return data
