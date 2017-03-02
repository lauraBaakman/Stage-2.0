# def configuration(parent_package='', top_path=None):
#     from numpy.distutils.misc_util import Configuration
#
#     config = Configuration('',
#                            parent_package,
#                            top_path)
#     config.add_extension(
#         '_kernels',  # Name of the extension
#         sources=['kernelsModule.c', 'kernels.c', '../utils.c']
#     )
#     return config
#
# if __name__ == "__main__":
#     from numpy.distutils.core import setup
#     setup(configuration=configuration, requires=['numpy', 'sklearn'])


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
