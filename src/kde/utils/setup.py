from distutils.core import setup, Extension

import setup_globals

sources = [
    'utilsModule.c',
    '../utils.c',
    'distancematrix.c',
    'knn.c',
    'covariancematrix.c',
    'eigenvalues.c',
    'geometricmean.c'
    ]

if __name__ == "__main__":
    module = Extension('_utils',
                       library_dirs=setup_globals.library_dirs,
                       libraries=setup_globals.libraries,
                       include_dirs=setup_globals.include_dirs,
                       sources=sources
                       )
    setup(name='_utils', version='1.0', ext_modules=[module])
