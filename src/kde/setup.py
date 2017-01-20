def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('',
                           parent_package,
                           top_path)
    config.add_extension(
        '_kde',  # Name of the extension
        sources=['utils.c', 'kde.c', 'parzen.c', 'kernels/densityFunctions.c']
    )
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration, requires=['numpy'])