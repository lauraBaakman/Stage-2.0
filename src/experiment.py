import platform
import argparse

from unipath import Path

import kde.utils.automaticWindowWidthMethods as automaticWindowWidthMethods
from kde.sambe import SAMBEstimator
from kde.mbe import MBEstimator
from kde.parzen import ParzenEstimator
from kde.kernels.epanechnikov import Epanechnikov
from kde.utils.grid import Grid
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
    # 'silverman': lambda d: 0.5,
    'breiman': lambda d: 1.0 / d
}

estimators = {
    'sambe': SAMBEstimator,
    'mbe': MBEstimator
}

_ask_for_confirmation = False
_use_xs_grid = False
_grid_dimension = 128


def handle_data_set(x_s, xi_s, data_set, data_set_file, *args):
        _, dimension = xi_s.shape
        general_bandwidth, results = estimate_pilot_densities(
            xi_s=xi_s,
            x_s=x_s
        )
        pilot_densities = results.densities
        if not _use_xs_grid:
            write(
                result=results,
                xs_out_path=ioUtils.build_x_result_data_path(
                    _results_path, data_set_file, 'parzen', *args
                ),
                xis_out_path=ioUtils.build_xi_result_data_path(
                    _results_path, data_set_file, 'parzen', *args
                )
            )

        for estimator_name, Estimator in estimators.items():
            print("\tEstimator: {}".format(estimator_name))

            for sensitivity_name, sensitivity_method in sensitivities.items():
                sensitivity = sensitivity_method(dimension)
                print("\t\tSensitivity: {value} ({method_name})".format(
                    value=sensitivity,
                    method_name=sensitivity_name)
                )

                result = run_single_configuration(
                    x_s=x_s,
                    xi_s=xi_s,
                    estimator=Estimator(
                                 dimension=dimension,
                                 sensitivity=sensitivity
                            ),
                    pilot_densities=pilot_densities,
                    general_bandwidth=general_bandwidth,
                    data_set=data_set
                )
                write(
                    result=result,
                    xs_out_path=ioUtils.build_x_result_data_path(
                        _results_path, data_set_file, estimator_name, sensitivity_name, *args
                    ),
                    xis_out_path=ioUtils.build_xi_result_data_path(
                        _results_path, data_set_file, estimator_name, sensitivity_name, *args
                    )
                )

        if _use_xs_grid:
            run_parzen(general_bandwidth, data_set, x_s, xi_s, dimension, data_set_file, *args)


def run_parzen(general_bandwidth, data_set, x_s, xi_s, dimension, data_set_file, *args):
    estimator_name = 'parzen'
    print("\tEstimator: {}".format('Parzen'))
    result = run_single_configuration(
        x_s=x_s,
        xi_s=xi_s,
        estimator=ParzenEstimator(
            dimension=dimension
        ),
        pilot_densities=None,
        general_bandwidth=general_bandwidth,
        data_set=data_set
    )
    write(
        result=result,
        xs_out_path=ioUtils.build_x_result_data_path(
            _results_path, data_set_file, estimator_name, *args
        ),
        xis_out_path=ioUtils.build_xi_result_data_path(
            _results_path, data_set_file, estimator_name, *args
        )
    )


def estimate_pilot_densities(x_s, xi_s):
    print("\tEstimating Pilot Densities")
    general_bandwidth = automaticWindowWidthMethods.ferdosi(xi_s)
    _, dimension = x_s.shape
    results = ParzenEstimator(
        dimension=dimension,
        bandwidth=general_bandwidth,
        kernel_class=Epanechnikov
    ).estimate(xi_s=x_s, x_s=xi_s)
    return general_bandwidth, results


def run_single_configuration(x_s, xi_s, estimator, pilot_densities, general_bandwidth, data_set):
    results = estimator.estimate(
        x_s=x_s, xi_s=xi_s,
        pilot_densities=pilot_densities, general_bandwidth=general_bandwidth
    )
    return results


def write(result, xs_out_path, xis_out_path=None):
    if xis_out_path is not None:
        with open(xs_out_path, 'wb') as x_out_file, open(xis_out_path, 'wb') as xi_out_file:
            result.to_file(x_out_file, xi_out_file)
    else:
        with open(xs_out_path, 'wb') as x_out_file:
            result.to_file(x_out_file)


def run_experiment_xs_is_xis(data_set_files):
    def handle_data_set_file(data_set_file, x_s=None):
        with open(data_set_file, 'r') as in_file:
            print("Data set: {}".format(data_set_file))
            data_set = io.DataSet.from_file(in_file=in_file)

            if not x_s:
                x_s = data_set.patterns

            handle_data_set(
                x_s=x_s, xi_s=data_set.patterns,
                data_set=data_set,
                data_set_file=data_set_file
            )

    for data_set_file in data_set_files:
        handle_data_set_file(data_set_file)


def run_experiment_with_xs_grid(data_set_files):
    def handle_data_set_file(data_set_file):
        with open(data_set_file, 'r') as in_file:
            print("Data set: {}".format(data_set_file))
            data_set = io.DataSet.from_file(in_file=in_file)
            xs = create_and_write_grid(data_set)
            handle_data_set(
                xs, data_set.patterns,
                None, data_set_file,
                'grid_{}'.format(_grid_dimension)
            )

    def create_and_write_grid(data_set):
        xs = Grid.cover(data_set.patterns, number_of_grid_points=_grid_dimension).grid_points
        grid_out_file = ioUtils.build_xs_path(_results_path, data_set_file, 'grid_{}'.format(_grid_dimension))
        with open(grid_out_file, 'w') as out_file:
            io.DataSet(patterns=xs).to_file(out_file)
        return xs

    for data_set_file in data_set_files:
        handle_data_set_file(data_set_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input-directory',
                        action=InputDirectoryAction,
                        default=_data_set_path,
                        help='The folder from which to read the data set files.')
    parser.add_argument('--datasets', default=list(), nargs='+',
                        help='A list of files with datasets.')
    parser.add_argument('-o', '--output-directory',
                        action=OutputDirectoryAction,
                        default=_results_path,
                        help='The folder to write the result files to.')
    parser.add_argument('--verify-files', dest='verify_files',
                        action='store_true',
                        help='For each dataset request confirmation before using it.')
    parser.add_argument('--no-verify-files', dest='verify_files',
                        action='store_false',
                        help='Do not request confirmation, use all dataset in the input directory.')
    parser.add_argument('--xs-grid', dest='use_xs_grid',
                        action='store_true',
                        help='''Compute the densities of the vertices of a grid that covers the data, '''
                             '''instead of for the data points.''')
    parser.add_argument('--no-xs-grid', dest='use_xs_grid',
                        action='store_false',
                        help='Compute the densities for the data points.')
    parser.add_argument(
        '-d', '--grid-dimension',
        type=int, default=_grid_dimension,
        help='''The number of vertices, in one dimenions, of the grid that covers the data, i.e. 5 results in '''
             '''a 5 x 5 x 5 grid if 3D data are used.'''
    )
    parser.set_defaults(verify_files=_ask_for_confirmation)
    parser.set_defaults(use_xs_grid=_use_xs_grid)
    args = parser.parse_args()

    _data_set_path = args.input_directory
    _results_path = args.output_directory
    _ask_for_confirmation = args.verify_files
    _use_xs_grid = args.use_xs_grid
    _grid_dimension = args.grid_dimension
    data_set_files = args.datasets

    if not data_set_files:
        data_set_files = ioUtils.get_data_set_files(_data_set_path, _ask_for_confirmation)

    if _use_xs_grid:
        print('Using a {} x ... x {} grid as xs.\n'.format(_grid_dimension, _grid_dimension))
        run_experiment_with_xs_grid(data_set_files)
    else:
        print('Using xis as xs.\n')
        run_experiment_xs_is_xis(data_set_files)
