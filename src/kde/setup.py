from distutils.core import setup, Extension

import numpy.distutils.misc_util as npdisutils


if __name__ == "__main__":
    include_dirs =['/Users/laura/.virtualenvs/stage/include/python3.5m',
                   '/Users/laura/.virtualenvs/stage/lib/python3.5/site-packages/numpy/core/include/numpy',
                   '/Users/laura/.virtualenvs/stage/include',
                   '/usr/local/include']
    include_dirs.extend(npdisutils.get_numpy_include_dirs())
    library_dirs = ['/Users/laura/.virtualenvs/stage/lib',
                    '/usr/local/lib']
    libraries = ['gsl',
                 'gslcblas',
                 'm']

    sources = ['utils.c', 'kdeModule.c', 'parzen.c', 'modifeidbreiman.c', 'kernels/kernels.c']

    module = Extension('_kde',
                       library_dirs= library_dirs,
                       libraries=libraries,
                       include_dirs=include_dirs,
                       sources=sources
                       )
    setup(name='_kde', version='1.0', ext_modules=[module])
