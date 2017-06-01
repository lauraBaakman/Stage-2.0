from distutils.core import setup, Extension

import setup_globals

sources = ['kernelsModule.c',
           'kernels.c',
           'gaussian.c',
           'epanechnikov.c',
           'testkernel.c',
           '../utils.c',
           '../utils/geometricmean.c',
           '../utils/eigenvalues.c',
           '../utils/gsl_utils.c']

if __name__ == "__main__":
    module = Extension('_kernels',
                       library_dirs=setup_globals.library_dirs,
                       libraries=setup_globals.libraries,
                       include_dirs=setup_globals.include_dirs,
                       sources=sources
                       )
    setup(name='_kernels', version='1.0', ext_modules=[module])
