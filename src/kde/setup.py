from distutils.core import setup, Extension

import setup_globals

import kernels

_utils_files = [
    'covariancematrix.c',
    'eigenvalues.c',
    'geometricmean.c',
    'gsl_utils.c',
    'knn.c',
]
_utils_path = 'utils'

sources = [
    'utils.c',
    'kdeModule.c',
    'mbe.c',
    'parzen.c',
    'sambe.c',
]

absolute_sources = [
  str(setup_globals.wd_path.child(source))
  for source
  in sources
]

absolute_sources.extend(kernels.files)

absolute_sources.extend([
    str(setup_globals.wd_path.child(_utils_path).child(u_file))
    for u_file in _utils_files
])
absolute_sources.append(
  setup_globals.lib_path.child('kdtree').child('kdtree.c')
)

absolute_sources = map(str, absolute_sources)

if __name__ == "__main__":
    module = Extension('_kde',
                       library_dirs=setup_globals.library_dirs,
                       libraries=setup_globals.libraries,
                       include_dirs=setup_globals.include_dirs,
                       sources=absolute_sources
                       )
    setup(name='_kde', version='1.0', ext_modules=[module])
