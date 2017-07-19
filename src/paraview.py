import argparse
import itertools
import logging

from unipath import Path
import numpy as np

from argparseActions import InputDirectoryAction, OutputDirectoryAction
import inputoutput.utils as ioUtils
import inputoutput.files as ioFiles
import inputoutput.dataset as ioDataset
import inputoutput.results as ioResults


_default_xs_path = Path('../data/simulated/small')
_default_densities_path = Path('../results/simulated/small')
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


def find_associated_result_files(xs_files, densities_files):
    skip_keys = ['file']
    for xs_file in xs_files:
        xs_file['associated results'] = filter(
            lambda x: ioFiles.is_associated_result_file(xs_file, x, skip_keys=skip_keys),
            densities_files
        )
    return xs_files


def process_files(files):
    for file in files:
        try:
            process_data_set_with_results(file)
        except Exception as e:
            logging.error(
                'An error occurred while processing the file {path}:\n {error}'.format(
                    path=file['file'],
                    error=e.message
                )
            )


def process_data_set_with_results(dataset_file):

    def should_be_created(path):
        if not path.exists():
            return True
        if args.replace_existing:
            logging.info('Replacing the file {}.'.format(path))
            return True
        else:
            logging.info('Skipping the generation of the file {}.'.format(path))
            return False

    def add_column_to_end(matrix, column):
        (_, last_idx) = matrix.shape
        return np.insert(matrix, last_idx, column, axis=1)

    def data_to_file(data, header, out_path):
        np.savetxt(
            fname=out_path,
            X=data,
            delimiter=", ",
            header=header,
            comments=''
        )
        logging.info('Writing to {}'.format(out_path))

    def determine_out_path(meta_data):
        def build_out_file(meta_data):
            if 'grid size' in meta_data.keys():
                return '{dataset_name}_grid_{grid_size}_paraview.csv'.format(
                    dataset_name=meta_data['semantic name'],
                    grid_size=meta_data['grid size']
                )
            else:
                return '{dataset_name}_paraview.csv'.format(
                    dataset_name=meta_data['semantic name'],
                )

        return args.output_directory.child(build_out_file(meta_data))

    def update_header(header, column_header):
        header = '{old_header}, {column_header}'.format(
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

    def add_estimated_densities(data_meta_data, header, data):
        results = dataset_file['associated results']
        estimated_densities = dict()
        for result in results:
            densities = ioResults.Results.from_file(result['file']).values
            estimated_densities[result['file']] = densities
            data = add_column_to_end(data, densities)
            header = update_header(header, estimator_description(result))
        return header, data, estimated_densities

    def add_square_errors(data_meta_data, header, data, estimated_densities):
        results = dataset_file['associated results']
        pairs = itertools.combinations(results, 2)
        for (a_meta, b_meta) in pairs:
            differences, column_header = add_square_error(
                a_meta=a_meta, a_values=estimated_densities[a_meta['file']],
                b_meta=b_meta, b_values=estimated_densities[b_meta['file']]
            )
            header = update_header(header, column_header)
            data = add_column_to_end(data, differences)
        return header, data

    out_path = determine_out_path(dataset_file)
    if not should_be_created(out_path):
        return

    logging.info(
        'Processing {xs_file}, with result files:{result_files}'.format(
            xs_file=ioUtils.partial_path(dataset_file['file']),
            result_files='\n' + '\n'.join([
                '\t{}'.format(ioUtils.partial_path(result_file['file']))
                for result_file
                in dataset_file['associated results']]
            )
        )
    )

    header, data = read_data_set_file(dataset_file['file'])
    header, data, estimated_densities = add_estimated_densities(data_meta_data=dataset_file, header=header, data=data)
    header, data = add_square_errors(
        data_meta_data=dataset_file,
        header=header, data=data,
        estimated_densities=estimated_densities
    )
    data = subsample(data, dataset_file)

    data_to_file(header=header, data=data, out_path=out_path)


def subsample(data, meta_data):
    def monte_carlo_subsample(data, probability):
        print('Monte carlo')
        return data

    def grid_subsample(data, offset):
        logging.error('The grid subsample implementation needs to be fixed.')
        return data[::offset + 1]

    if 'grid size' in meta_data:
        return grid_subsample(data, args.sub_sampling_space)
    else:
        return monte_carlo_subsample(data, args.sub_sampling_probability)


def estimator_description(meta_data):
    return '{estimator}{sensitivity}'.format(
        estimator=meta_data['estimator'],
        sensitivity=' ({})'.format(meta_data['sensitivity']) if meta_data['sensitivity'] else ''
    )


def add_square_error(a_meta, a_values, b_meta, b_values):
    def square_error(a_values, b_values):
        return np.power(a_values - b_values, 2)

    def determine_column_header(a_meta, b_meta):
        return 'squared[{a} - {b}]'.format(
            a=estimator_description(a_meta),
            b=estimator_description(b_meta)
        )

    square_error = square_error(a_values, b_values)
    column_header = determine_column_header(a_meta, b_meta)
    return square_error, column_header


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR)
    logging.info('Running in verbose mode')
    logging.info('Reading xs files from: {}'.format(args.xs_directory))
    logging.info('Reading density files from: {}'.format(args.densities_directory))

    files = find_associated_result_files(
        xs_files=collect_meta_data(ioUtils.get_data_set_files(args.xs_directory, show_files=False)),
        densities_files=collect_meta_data(ioUtils.get_result_files(args.densities_directory, show_files=False))
    )
    process_files(files)
