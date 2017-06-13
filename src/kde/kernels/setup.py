import sys
from unipath import Path
sys.path.append(Path(__file__).ancestor(2))

from distutils.core import setup, Extension

import setup_globals

_utils_files = [
  'geometricmean.c',
  'eigenvalues.c',
  'gsl_utils.c'
]

_kde_files = [
  'utils.c',
]

_sources = [
  'kernelsModule.c',
  'kernels.c',
  'gaussian.c',
  'epanechnikov.c',
  'testkernel.c'
]

absolute_sources = map(
  lambda x: setup_globals.wd_path.child('kernels').child(x),
  _sources
)
absolute_sources.extend(map(
  lambda x: setup_globals.wd_path.child('utils').child(x),
  _utils_files
))
absolute_sources.extend(map(
  lambda x: setup_globals.wd_path.child(x),
  _kde_files
))

absolute_sources = map(str, absolute_sources)

if __name__ == "__main__":
    module = Extension('_kernels',
                       library_dirs=setup_globals.library_dirs,
                       libraries=setup_globals.libraries,
                       include_dirs=setup_globals.include_dirs,
                       sources=absolute_sources
                       )
    setup(name='_kernels', version='1.0', ext_modules=[module])
