from distutils.core import setup, Extension

import globalConfigParameters

sources = ['kernelsModule.c', 'kernels.c', '../utils.c']

if __name__ == "__main__":
    module = Extension('_kernels',
                       library_dirs=globalConfigParameters.library_dirs,
                       libraries=globalConfigParameters.libraries,
                       include_dirs=globalConfigParameters.include_dirs,
                       sources=sources
                       )
    setup(name='_kernels', version='1.0', ext_modules=[module])
