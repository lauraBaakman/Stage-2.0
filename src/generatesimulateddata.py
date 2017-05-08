import os.path

import datasets.simulated

if __name__ == '__main__':
    extension = '.txt'
    path = '../data/simulated'

    for name, generator in datasets.simulated.generators.items():
        data_set = generator()
        path = os.path.join(path, name + extension)
        data_set.to_file(path)
