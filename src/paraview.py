import argparse
import logging

from unipath import Path
import numpy as np
import coloredlogs

from argparseActions import InputDirectoryAction, OutputDirectoryAction
import inputoutput.utils as ioUtils
import inputoutput.files as ioFiles
import inputoutput.dataset as ioDataset
import inputoutput.results as ioResults


_default_xs_path = Path('../data/simulated/small')
_default_densities_path = Path('../results/small/breiman')
_default_output_path = Path('.')
_default_subsample_probability = 0.01
_default_subsample_space = 31

args = None


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-x', '--xs-directory',
                        action=InputDirectoryAction,
                        default=_default_xs_path,
                        help='''The folder from which to read the files with the points for which the '''
                             '''densities were estimated.''')
    parser.add_argument('-i', '--densities-directory',
                        action=InputDirectoryAction,
                        default=_default_densities_path,
                        help='''The folder from which to read the files with the estimated densities.''')
    parser.add_argument('-o', '--output-directory',
                        action=OutputDirectoryAction,
                        default=_default_output_path,
                        help='The folder to write the paraview files to.')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help="increase output verbosity")
    parser.add_argument('-r', '--replace-existing',
                        action='store_true',
                        default=False,
                        help="Overwrite existing files in the output directory.")
    parser.add_argument('-s', '--sub-sample',
                        action='store_true',
                        default=False,
                        help="""If set to true the data is subsampled the data, the method depends on the """
                             """kind of data.""")
    parser.add_argument('--sub-sampling-probability',
                        type=float, default=_default_subsample_probability,
                        help="The subsampling probability if monte carlo subsampling is used.")
    parser.add_argument('--sub-sampling-space',
                        type=int, default=_default_subsample_space,
                        help="""The number of elements to skip between two subsequent elements that are """
                             """included in the sub sampled version of the data set.""")
    return parser


def collect_meta_data(paths):
    files = list()
    for path in paths:
        meta = ioFiles.parse_path(path)
        meta.update({'file': path})
        files.append(meta)
    return files


def find_associated_result_files(xs_files, xs_results_files, xis_results_files=None):
    xis_results_files = xis_results_files or list()
    skip_keys = ['file']
    for xs_file in xs_files:
        xs_file['xs result files'] = filter(
            lambda x: ioFiles.is_associated_result_file(xs_file, x, skip_keys=skip_keys),
            xs_results_files
        )
        for xs_result_file in xs_file['xs result files']:
            associated_xis_files = filter(
                lambda x: ioFiles.is_associated_xis_file(xs_result_file, x, skip_keys=skip_keys),
                xis_results_files
            )
            if associated_xis_files:
                xs_result_file['xis result file'] = associated_xis_files.pop()
    return xs_files


def process_files(files):
    for file in files:
        try:
            process_xs_data(file)
            process_xis_data(file)
        except Exception as e:
            logging.exception(
                'An error occurred while processing the file {path}:'.format(
                    path=file['file'],
                )
            )


def process_xs_data(dataset_file):

    def add_column_to_end(matrix, column):
        (_, last_idx) = matrix.shape
        return np.insert(matrix, last_idx, column, axis=1)

    def update_header(header, column_header):
        header = '{old_header},{column_header}'.format(
            old_header=header,
            column_header=column_header
        )
        return header

    def read_data_set_file(dataset_file):
        data_set = ioDataset.DataSet.from_file(dataset_file)
        data = data_set.patterns
        header = 'x, y, z'

        if data_set.has_densities:
            data = add_column_to_end(data, data_set.densities)
            header = update_header(header, 'true densities')

        return header, data

    def add_configuration_specific_data(data_meta_data, header, data):
        results_meta_data = dataset_file['xs result files']
        estimated_densities = dict()
        for result_meta_data in results_meta_data:
            result = ioResults.Results.from_file(result_meta_data['file'])

            densities = result.densities
            estimated_densities[result_meta_data['file']] = densities

            try:
                estimator = estimator_description(result_meta_data)

                data = add_column_to_end(data, densities)
                header = update_header(header, estimator)

                data = add_column_to_end(data, result.num_used_patterns)
                header = update_header(
                    header,
                    'num used patterns ({estimator})'.format(estimator=estimator)
                )
            except Exception as e:
                logging.exception(
                    'An error occurred while processing the results file {path}:\n {error}\nTraceBack:'.format(
                        path=result_meta_data['file'],
                        error=e.message,
                    )
                )
        return header, data, estimated_densities

    out_path = determine_out_path(dataset_file)
    if not should_path_be_created(out_path):
        return

    log_processing(
        file_type_string='xs result files',
        result_files=dataset_file['xs result files'],
        dataset_file=dataset_file['file']
    )

    header, data = read_data_set_file(dataset_file['file'])
    header, data, estimated_densities = add_configuration_specific_data(
        data_meta_data=dataset_file, header=header, data=data
    )
    data = subsample(data, dataset_file)

    data_to_file(header=header, data=data, out_path=out_path)


def process_xis_data(dataset_meta):
    def read_baseline_xis(dataset_meta):
        if not dataset_meta['xs result files']:
            return None, None

        xs_result_meta = dataset_meta.get('xs result files')[0]
        xi_file_meta = xs_result_meta.get('xis result file', dict()).get('file')

        if not xi_file_meta:
            return None, None

        result = ioResults.Results.from_file(
            x_file=xs_result_meta['file'],
            xi_file=xi_file_meta
        )
        return (
            result.xis,
            ['xis_x', 'xis_y', 'xis_z']
        )

    def create_column_name(base_name, meta_data):
        return '_'.join([base_name, meta_data['estimator'], meta_data.get('sensitivity', '')])

    def collect_xis_data_from_file(baseline_xis, xs_result_file_meta, old_data, header):
        result = ioResults.Results.from_file(
            x_file=xs_result_file_meta['file'],
            xi_file=xs_result_file_meta.get('xis result file', dict()).get('file')
        )

        # Check if result.xis equals xis
        if not np.array_equal(baseline_xis, result.xis):
            raise UserWarning(
                'The xis of the base line file, and {current_xis_file} are not the same.'.format(
                    current_xis_file=xs_result_file_meta.get('xis result file', dict()).get('file')
                )
            )

        # Get the xis data from result
        matrix, partial_header = result.xis_data_matrix
        matrix, partial_header = matrix[:, 3:], partial_header[3:]

        # Add prefix to the partial_header
        partial_header = [create_column_name(base_name, xs_result_file_meta) for base_name in partial_header]

        # Add (matrix, partial_header) to (data, header)
        new_data = np.hstack((old_data, matrix))
        header.extend(partial_header)

        return new_data, header

    def collect_all_xis_data(dataset_meta):
        baseline_xis, header = read_baseline_xis(dataset_meta)

        if not baseline_xis:
            return None, None

        data = baseline_xis
        for xs_result_file_meta in dataset_meta['xs result files']:
            data, header = collect_xis_data_from_file(
                xs_result_file_meta=xs_result_file_meta,
                baseline_xis=baseline_xis,
                old_data=data, header=header
            )
        return data, header

    out_path = determine_out_path(dataset_meta, appendices=['xis'])
    if not should_path_be_created(out_path):
        return
    log_processing(
        file_type_string='xis result files',
        result_files=[
            xs_result_file.get('xis result file', None)
            for xs_result_file
            in dataset_meta.get('xs result files', list())
        ],
        dataset_file=dataset_meta['file']
    )

    data, column_names = collect_all_xis_data(dataset_meta)

    logging.info('Subsampling the data gathered with the base file {xsfile}.'.format(
        xsfile=ioUtils.partial_path(dataset_meta['file'])
    ))

    data = subsample(data, dataset_meta)
    header = ', '.join(column_names)

    data_to_file(data, header, out_path)


def log_processing(file_type_string, result_files, dataset_file):
    def result_files_are_none():
        return bool(filter(lambda x: x is None, result_files))

    if result_files_are_none():
        message = 'No files of {file_type} to process for {xs_file}'.format(
                file_type=file_type_string,
                xs_file=ioUtils.partial_path(dataset_file)
            )
    else:
        message = 'Processing {xs_file}, with {file_type}:{result_files}'.format(
                xs_file=ioUtils.partial_path(dataset_file),
                result_files='\n' + '\n'.join([
                    '\t{}'.format(ioUtils.partial_path(result_file['file']))
                    for result_file
                    in result_files]
                ),
                file_type=file_type_string
            )
    logging.info(message)


def should_path_be_created(path):
    if not path.exists():
        return True
    if args.replace_existing:
        logging.info('Replacing the file {}.'.format(path))
        return True
    else:
        logging.info('Skipping the generation of the file {}.'.format(path))
        return False


def data_to_file(data, header, out_path):
    np.savetxt(
        fname=out_path,
        X=data,
        delimiter=", ",
        header=header,
        comments='',
        fmt='%0.15f'
    )
    logging.info('Writing to {}'.format(out_path))


def determine_out_path(meta_data, appendices=None):
    def build_out_file(semantic_name, *args):
        args_string = '_'.join(args)
        return '{data_set}{seperator}{args_string}_paraview.csv'.format(
            data_set=semantic_name,
            seperator='_' if args_string else '',
            args_string=args_string
        )

    appendices = appendices if appendices else list()
    if 'grid_size' in meta_data.keys():
        appendices.append('grid')
        appendices.append(str(meta_data['grid_size']))
    if args.sub_sample:
        appendices.append('subsampled')
    return args.output_directory.child(build_out_file(meta_data['semantic_name'], *appendices))


def subsample(data, meta_data):
    def monte_carlo_subsample(data, probability):
        num_data_points, _ = data.shape
        return data[np.random.sample(num_data_points) < probability, :]

    def grid_subsample(data, offset):
        return ioUtils.sub_sample_grid(data, offset)

    if not args.sub_sample:
        return data

    if 'grid_size' in meta_data:
        return grid_subsample(data, args.sub_sampling_space)
    else:
        return monte_carlo_subsample(data, args.sub_sampling_probability)


def estimator_description(meta_data):
    return '{estimator}{sensitivity}'.format(
        estimator=meta_data['estimator'],
        sensitivity=' ({})'.format(meta_data['sensitivity']) if meta_data['sensitivity'] else ''
    )


def add_error(a_meta, a_values, b_meta, b_values):
    def compute_error(a_values, b_values):
        return a_values - b_values

    def determine_column_header(a_meta, b_meta):
        return '{a} - {b}'.format(
            a=estimator_description(a_meta),
            b=estimator_description(b_meta)
        )

    error = compute_error(a_values, b_values)
    column_header = determine_column_header(a_meta, b_meta)
    return error, column_header


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    coloredlogs.install()
    logging.getLogger().setLevel(logging.INFO if args.verbose else logging.ERROR)
    logging.info('Running in verbose mode')
    logging.info('Reading xs files from: {}'.format(args.xs_directory))
    logging.info('Reading density files from: {}'.format(args.densities_directory))

    files = find_associated_result_files(
        xs_files=collect_meta_data(ioUtils.get_data_set_files(args.xs_directory, show_files=False)),
        xs_results_files=collect_meta_data(ioUtils.get_xs_result_files(args.densities_directory, show_files=False)),
        xis_results_files=collect_meta_data(ioUtils.get_xis_result_files(args.densities_directory, show_files=False)),
    )
    process_files(files)
