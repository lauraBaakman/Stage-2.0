import argparse
import os

from unipath import Path

from kde.sambe import SAMBEstimator
from kde.modifeidbreiman import MBEstimator
import inputoutput

_data_set_path = Path(
    '/Users/laura/Repositories/stage-2.0/data/simulated/small/')
_results_path = Path(
    '/Users/laura/Repositories/stage-2.0/results/simulated/small/')

sensitivities = {
    'silverman': lambda d: 0.5,
    'breiman': lambda d: 1.0 / d
}

estimators = {
    'sambe': SAMBEstimator,
    'mbe': MBEstimator
}


ask_for_confirmation = False


def partial_path(path):
    return Path(path.components()[-3:])


def get_data_set_files(input_path):

    def add_data_set(data_set):
        files.append(data_set)

    def skip_data_set(*args, **kwargs):
        pass

    potential_data_sets = input_path.walk(filter=lambda x: x.ext == '.txt')
    files = list()

    responses = {
        'y': add_data_set,
        'n': skip_data_set
    }

    if ask_for_confirmation:
        for data_set in potential_data_sets:
            response = input(
                'Include the data set in the file ..{}? [y/N]\n'.format(
                    partial_path(data_set))
            ).lower()
            responses.get(response, skip_data_set)(data_set)
        return files
    else:
        files.extend(potential_data_sets)
        print("Running the experiment on these datasets:\n{data_sets}\n".
              format(
                     data_sets='\n'.join([
                                         partial_path(file)
                                         for file
                                         in files])
                     )
              )
        return files


def handle_dataset(data_set):
    for estimator_name, Estimator in estimators.items():
        print("\tEstimator: {}".format(estimator_name))

        for sensitivity_name, sensitivity_method in sensitivities.items():
            sensitivity = sensitivity_method(data_set.dimension)
            print("\t\tSensitivity: {value} ({method_name})".format(
                value=sensitivity,
                method_name=sensitivity_name)
            )

            result = run(data_set,
                         Estimator(
                             dimension=data_set.dimension,
                             sensitivity=sensitivity))

            out_path = build_output_path(
                                         data_set_file,
                                         estimator_name,
                                         sensitivity_name)
            with open(out_path, 'a') as out_file:
                result.to_file(out_file)


def run(data_set, estimator):
    densities = estimator.estimate(data_set.patterns, data_set.patterns)
    return inputoutput.Results(densities, data_set)


def build_output_path(data_set_file, estimator, sensitivity):
    out_file_name = '{data_set}_{estimator}_{sensitivity}.txt'.format(
        data_set=Path(data_set_file).stem,
        estimator=estimator,
        sensitivity=sensitivity
    )
    return _results_path.child(out_file_name)


class InputDirectoryAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = Path(values).absolute()
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError(
                "{0} is not a valid path".format(prospective_dir)
            )
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError(
                "{0} is not a readable dir".format(prospective_dir)
            )


class OutputDirectoryAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = Path(values).absolute()
        prospective_dir.mkdir(parents=True)
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError(
                "{0} is not a valid path".format(prospective_dir)
            )
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError(
                "{0} is not a readable dir".format(prospective_dir)
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory',
                        action=InputDirectoryAction,
                        default=_data_set_path)
    parser.add_argument('-o', '--output_directory',
                        action=OutputDirectoryAction,
                        default=_results_path)
    args = parser.parse_args()

    data_set_path = args.input_directory
    _results_path = args.output_directory

    data_set_files = get_data_set_files(data_set_path)

    for data_set_file in data_set_files:
        with open(data_set_file, 'r') as in_file:
            print("Data set: {}".format(data_set_file))
            data_set = inputoutput.DataSet.from_file(in_file=data_set_file)
            handle_dataset(data_set)
