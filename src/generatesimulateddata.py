import os.path
import argparse

import datasets.simulated.allsets
from argparseActions import OutputDirectoryAction

scale = 1.0
extension = '.txt'
_output_path = '/Users/laura/Repositories/stage/data/simulated'


def generate_data_set(name, generator):
    data_set = generator(scale)
    out_path = os.path.join(_output_path, name.replace(" ", "_") + "_" + str(data_set.num_patterns) + extension)
    with open(out_path, 'wb') as out_file:
        data_set.to_file(out_file)


def run():
    data_sets = datasets.simulated.allsets.sets

    for name, generator in data_sets.items():
        print('Generating {} ...'.format(name))
        generate_data_set(name, generator)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-o', '--output_directory',
        action=OutputDirectoryAction, default=_output_path,
        help='The directory to store the generated datasets.'
    )
    parser.add_argument(
        '-s', '--scale',
        type=float, default=scale,
        help="""The scale of the number of elements in the datasets, if the scale is 1.0 the sizes as """
             """defined by Ferdosi et al. are used."""
    )
    args = parser.parse_args()

    _output_path = args.output_directory
    scale = args.scale

    run()
