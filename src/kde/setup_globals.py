from unipath import Path
import numpy.distutils.misc_util as npdisutils
from environs import Env

_environment = Env()

_virtual_env_path = Path(_environment('VIRTUALENVPATH'))
_local_path = Path(_environment('LOCALPATH', '/usr/local'))

include_dirs = [
    _virtual_env_path.child('include').child('python3.5m'),
    _virtual_env_path.child('include'),
    _local_path.child('include')
]
include_dirs.extend(npdisutils.get_numpy_include_dirs())

library_dirs = [
    _virtual_env_path.child('lib'),
    _local_path.child('lib')
]

libraries = [
    'gsl',
    'gslcblas',
    'm'
]

lib_path = Path(__file__).ancestor(2).child('lib')

wd_path = Path(__file__).ancestor(1)
