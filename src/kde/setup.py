from distutils.core import setup, Extension

import setup_globals

sources = [
    'kdeModule.c',
    'modifeidbreiman.c',
    'parzen.c',
    'sambe.c',
    'utils.c',
    'kernels/kernels.c',
    'kernels/epanechnikov.c',
    'kernels/gaussian.c',
    'kernels/testkernel.c',
    'utils/covariancematrix.c',
    'utils/distancematrix.c',
    'utils/eigenvalues.c',
    'utils/geometricmean.c',
    'utils/gsl_utils.c',
    'utils/knn.c',
]

if __name__ == "__main__":
    module = Extension('_kde',
                       library_dirs=setup_globals.library_dirs,
                       libraries=setup_globals.libraries,
                       include_dirs=setup_globals.include_dirs,
                       sources=sources
                       )
    setup(name='_kde', version='1.0', ext_modules=[module])
