def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('',
                           parent_package,
                           top_path)
    config.add_extension(
        '_utils',  # Name of the extension
        sources=['utilsModule.c', '../utils.c', 'distancematrix.c', 'knn.c']
    )
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration, requires=['numpy'])
