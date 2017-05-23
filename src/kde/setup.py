from distutils.core import setup, Extension

import globalConfigParameters

sources = ['utils.c',
           'kdeModule.c',
           'parzen.c',
           'modifeidbreiman.c',
           'kernels/kernels.c',
           'sambe.c',
           'utils/distancematrix.c',
           'utils/knn.c',
           'utils/covariancematrix.c']

if __name__ == "__main__":
    module = Extension('_kde',
                       library_dirs=globalConfigParameters.library_dirs,
                       libraries=globalConfigParameters.libraries,
                       include_dirs=globalConfigParameters.include_dirs,
                       sources=sources
                       )
    setup(name='_kde', version='1.0', ext_modules=[module])
