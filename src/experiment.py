import platform
import argparse

from unipath import Path

import kde.utils.automaticWindowWidthMethods as automaticWindowWidthMethods
from kde.sambe import SAMBEstimator
from kde.mbe import MBEstimator
from kde.parzen import ParzenEstimator
from kde.kernels.epanechnikov import Epanechnikov
import inputoutput as io
import inputoutput.utils as ioUtils
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


def handle_dataset(data_set):
    general_bandwidth, pilot_densities = estimate_pilot_densities(xi_s=data_set.patterns, x_s=data_set.patterns)
    write(
        result=io.Results(pilot_densities, data_set),
        out_path=ioUtils.build_result_path(
            _results_path, data_set_file, 'parzen',
        )
    )

    for estimator_name, Estimator in estimators.items():
        print("\tEstimator: {}".format(estimator_name))

        for sensitivity_name, sensitivity_method in sensitivities.items():
            sensitivity = sensitivity_method(data_set.dimension)
            print("\t\tSensitivity: {value} ({method_name})".format(
                value=sensitivity,
                method_name=sensitivity_name)
            )

            result = run_single_configuration(
                data_set=data_set,
                estimator=Estimator(
                             dimension=data_set.dimension,
                             sensitivity=sensitivity
                        ),
                pilot_densities=pilot_densities,
                general_bandwidth=general_bandwidth
            )
            write(
                result=result,
                out_path=ioUtils.build_result_path(
                    _results_path, data_set_file, estimator_name, sensitivity_name
                )
            )


def estimate_pilot_densities(x_s, xi_s):
    print("\tEstimator: Parzen")
    general_bandwidth = automaticWindowWidthMethods.ferdosi(xi_s)
    _, dimension = x_s.shape
    pilot_densities = ParzenEstimator(
        dimension=dimension,
        bandwidth=general_bandwidth,
        kernel_class=Epanechnikov
    ).estimate(xi_s=x_s, x_s=xi_s)
    return general_bandwidth, pilot_densities


def run_single_configuration(data_set, estimator, pilot_densities, general_bandwidth):
    densities = estimator.estimate(
        x_s=data_set.patterns, xi_s=data_set.patterns,
        pilot_densities=pilot_densities, general_bandwidth=general_bandwidth
    )
    return io.Results(densities, data_set)


def write(result, out_path):
    with open(out_path, 'wb') as out_file:
        result.to_file(out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input_directory',
                        action=InputDirectoryAction,
                        default=_data_set_path,
                        help='The folder from which to read the data set files.')
    parser.add_argument('--datasets', default=list(), nargs='+',
                        help='A list of files with datasets.')
    parser.add_argument('-o', '--output_directory',
                        action=OutputDirectoryAction,
                        default=_results_path,
                        help='The folder to write the result files to.')
    parser.add_argument('--verify-files', dest='verify_files',
                        action='store_true',
                        help='For each dataset request confirmation before using it.')
    parser.add_argument('--no-verify-files', dest='verify_files',
                        action='store_false',
                        help='Do not request confirmation, use all dataset in the input directory.')
    parser.set_defaults(verify_files=_ask_for_confirmation)
    args = parser.parse_args()

    data_set_path = args.input_directory
    _results_path = args.output_directory
    _ask_for_confirmation = args.verify_files
    data_set_files = args.datasets

    if not data_set_files:
        data_set_files = ioUtils.get_data_set_files(data_set_path)

    for data_set_file in data_set_files:
        with open(data_set_file, 'r') as in_file:
            print("Data set: {}".format(data_set_file))
            data_set = io.DataSet.from_file(in_file=data_set_file)
            handle_dataset(data_set)
