import numpy.distutils.misc_util as npdisutils


include_dirs = ['/Users/laura/.virtualenvs/stage/include/python3.5m',
                '/Users/laura/.virtualenvs/stage/lib/python3.5/site-packages/numpy/core/include/numpy',
                '/Users/laura/.virtualenvs/stage/include',
                '/usr/local/include']
include_dirs.extend(npdisutils.get_numpy_include_dirs())

library_dirs = ['/Users/laura/.virtualenvs/stage/lib',
                '/usr/local/lib']

libraries = ['gsl',
             'gslcblas',
             'm']