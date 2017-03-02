from distutils.core import setup, Extension

import globalConfigParameters

sources = ['utilsModule.c', '../utils.c', 'distancematrix.c', 'knn.c', 'covariancematrix.c']

if __name__ == "__main__":
    module = Extension('_utils',
                       library_dirs=globalConfigParameters.library_dirs,
                       libraries=globalConfigParameters.libraries,
                       include_dirs=globalConfigParameters.include_dirs,
                       sources=sources
                       )
    setup(name='_utils', version='1.0', ext_modules=[module])
