import os.path

import datasets.simulated.allsets

if __name__ == '__main__':
    extension = '.txt'
    path = '/Users/laura/Repositories/stage-2.0/data/simulated'

    for name, generator in datasets.simulated.allsets.sets.items():
        print('Generating {} ...'.format(name))
        data_set = generator()
        out_path = os.path.join(path, name.replace(" ", "_") + "_" + str(data_set.num_patterns) + extension)
        with open(out_path, 'wb') as out_file:
            data_set.to_file(out_file)
