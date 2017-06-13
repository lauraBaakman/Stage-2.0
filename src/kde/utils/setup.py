import sys
from unipath import Path
sys.path.append(Path(__file__).ancestor(2))

from distutils.core import setup, Extension

import setup_globals

_sources = [
  'utilsModule.c',
  'gsl_utils.c',
  'knn.c',
  'covariancematrix.c',
  'eigenvalues.c',
  'geometricmean.c',
]

_kde_sources = [
    'utils.c',
]

absolute_sources = map(
  lambda x: setup_globals.wd_path.child('utils').child(x),
  _sources
)
absolute_sources.extend(map(
  lambda x: setup_globals.wd_path.child(x),
  _kde_sources
))
absolute_sources.append(
  setup_globals.lib_path.child('kdtree').child('kdtree.c')
)

absolute_sources = map(str, absolute_sources)

if __name__ == "__main__":
    module = Extension('_utils',
                       library_dirs=setup_globals.library_dirs,
                       libraries=setup_globals.libraries,
                       include_dirs=setup_globals.include_dirs,
                       sources=absolute_sources
                       )
    setup(name='_utils', version='1.0', ext_modules=[module])
