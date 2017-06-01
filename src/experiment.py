import numpy as np

from unipath import Path

from kde.sambe import SAMBEstimator
from kde.modifeidbreiman import MBEstimator
from inputoutput.dataset import DataSet
from inputoutput.results import Results

sensitivities = {
    'silverman': lambda d: 0.5,
    'breiman': lambda d: 1.0 / d
}

estimators = {
    'sambe': SAMBEstimator,
    'mbe': MBEstimator
}

data_set_files = [
    'ferdosi_1_600.txt',
    'ferdosi_1_60000.txt',
    'ferdosi_2_60000.txt',
    'ferdosi_3_120000.txt',
    'ferdosi_4_60000.txt',
    'ferdosi_5_60000.txt',
]

def build_input_path(dataset_file):
    raise NotImplementedError()

def handle_dataset(data_set):
    for estimator_name, Estimator in estimators.items():
        for sensitivity_name, sensitivity in sensitivities.items():
            result = run(data_set,
                         Estimator(
                             dimension=data_set.dimension,
                             sensitivity=sensitivity))

            out_path = build_output_path(data_set_file, estimator_name, sensitivity_name)
            with open(out_path, 'w') as out_file:
                result.to_file(out_file)


def run(data_set, estimator):
    densities = np.zeros(data_set.num_patterns)
    # densities = estimator.estimate(data_set.patterns, data_set.patterns)
    return Results(densities, data_set)


def build_output_path(dataset_file, estimator, sensitivity):
    raise NotImplementedError()


if __name__ == '__main__':
    for data_set_file in data_set_files:
        in_path = build_input_path(data_set_file)
        with open(in_path, 'r') as in_file:
            data_set = DataSet.from_file(in_file=in_path)
            handle_dataset(data_set)