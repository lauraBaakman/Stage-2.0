from distutils.core import setup, Extension

import numpy.distutils.misc_util as npdisutils


# def configuration(parent_package='', top_path=None):
#     config = Configuration('',
#                            parent_package,
#                            top_path)
#     config.add_extension(
#         '_kde',  # Name of the extension
#         sources=['utils.c', 'kdeModule.c', 'parzen.c', 'modifeidbreiman.c', 'kernels/kernels.c']
#     )
#     return config

if __name__ == "__main__":
    # from numpy.distutils.core import setup
    # setup(configuration=configuration, requires=['numpy'])

    # Source: https://github.com/ezbc/class_work/blob/261e1545aabcaabbfe1e1922b5e198d45cfa88c3/classes/machine_learning/project/gausspy_module/gausspy/tvc/code.c/setup.py

    module = Extension('_kde',
                       library_dirs='/Users/laura/.virtualenvs/stage/lib',
                       include_dirs=['/Users/laura/.virtualenvs/stage/include/python3.5m',
                                     '/Users/laura/.virtualenvs/stage/lib/python3.5/site-packages/numpy/core/include'
                                     '/Users/laura/.virtualenvs/stage/include',
                                     npdisutils.get_numpy_include_dirs()],
                       sources=['utils.c',
                                'kdeModule.c',
                                'parzen.c',
                                'modifeidbreiman.c',
                                'kernels/kernels.c']
                       )
    setup(name='_kde', version='1.0', ext_modules=[module])
