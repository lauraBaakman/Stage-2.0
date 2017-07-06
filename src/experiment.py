import platform
import argparse

from unipath import Path

import kde.utils.automaticWindowWidthMethods as automaticWindowWidthMethods
from kde.sambe import SAMBEstimator
from kde.mbe import MBEstimator
from kde.parzen import ParzenEstimator
from kde.kernels.epanechnikov import Epanechnikov
import inputoutput
from argparseActions import InputDirectoryAction, OutputDirectoryAction

if platform.system() == 'Darwin':
    _data_set_path = Path('/Users/laura/Repositories/stage/data/simulated')
    _results_path = Path('/Users/laura/Repositories/stage/results/simulated')
elif platform.system() == 'Linux':
    _data_set_path = Path('/home/laura/Stage-2.0/data/simulated')
    _results_path = Path('/home/laura/Stage-2.0/results/simulated')
else:
    print('No default paths supported for this system, use -i and -o')

sensitivities = {
    'silverman': lambda d: 0.5,
    'breiman': lambda d: 1.0 / d
}

estimators = {
    'sambe': SAMBEstimator,
    'mbe': MBEstimator
}

_ask_for_confirmation = False


def partial_path(path):
    return Path(path.components()[-3:])


def get_data_set_files(input_path):
    files = list(input_path.walk(filter=lambda x: x.ext == '.txt'))
    if _ask_for_confirmation:
        files = confirm_files(files)
    show_files_to_user(files)
    return files


def confirm_files(potential_files):
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


def show_files_to_user(files):
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


def handle_dataset(data_set):
    # Compute Pilot Densities
    print("\tEstimator: Parzen")
    general_bandwidth = automaticWindowWidthMethods.ferdosi(data_set.patterns)
    pilot_densities = ParzenEstimator(
        dimension=data_set.dimension,
        bandwidth=general_bandwidth,
        kernel_class=Epanechnikov
    ).estimate(xi_s=data_set.patterns, x_s=data_set.patterns)
    result = inputoutput.Results(pilot_densities, data_set)
    write(result, data_set_file, estimator_name='parzen')

    for estimator_name, Estimator in estimators.items():
        print("\tEstimator: {}".format(estimator_name))

        for sensitivity_name, sensitivity_method in sensitivities.items():
            sensitivity = sensitivity_method(data_set.dimension)
            print("\t\tSensitivity: {value} ({method_name})".format(
                value=sensitivity,
                method_name=sensitivity_name)
            )

            result = run(
                data_set=data_set,
                estimator=Estimator(
                             dimension=data_set.dimension,
                             sensitivity=sensitivity
                        ),
                pilot_densities=pilot_densities,
                general_bandwidth=general_bandwidth
            )
            write(result, data_set_file, estimator_name, sensitivity_name)


def run(data_set, estimator, pilot_densities, general_bandwidth):
    densities = estimator.estimate(
        x_s=data_set.patterns, xi_s=data_set.patterns,
        pilot_densities=pilot_densities, general_bandwidth=general_bandwidth
    )
    return inputoutput.Results(densities, data_set)


def write(result, data_set_file, estimator_name, sensitivity_name=None):
            out_path = build_output_path(data_set_file,
                                         estimator_name,
                                         sensitivity_name)
            with open(out_path, 'wb') as out_file:
                result.to_file(out_file)


def build_output_path(data_set_file, estimator, sensitivity):
    data_set_estimator_part = '{data_set}_{estimator}'.format(
        data_set=Path(data_set_file).stem,
        estimator=estimator,
    )
    if sensitivity:
        out_file_name = '{data_set_estimator}_{sensitivity}.txt'.format(
            data_set_estimator=data_set_estimator_part,
            sensitivity=sensitivity
        )
    else:
        out_file_name = '{data_set_estimator}.txt'.format(
            data_set_estimator=data_set_estimator_part
        )
    return _results_path.child(out_file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input_directory',
                        action=InputDirectoryAction,
                        default=_data_set_path)
    parser.add_argument('-o', '--output_directory',
                        action=OutputDirectoryAction,
                        default=_results_path)
    parser.add_argument('--verify-files', dest='verify_files',
                        action='store_true')
    parser.add_argument('--no-verify-files', dest='verify_files',
                        action='store_false')
    parser.set_defaults(verify_files=_ask_for_confirmation)
    args = parser.parse_args()

    data_set_path = args.input_directory
    _results_path = args.output_directory
    _ask_for_confirmation = args.verify_files

    data_set_files = get_data_set_files(data_set_path)

    for data_set_file in data_set_files:
        with open(data_set_file, 'r') as in_file:
            print("Data set: {}".format(data_set_file))
            data_set = inputoutput.DataSet.from_file(in_file=data_set_file)
            handle_dataset(data_set)
