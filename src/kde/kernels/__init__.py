from unipath import Path

_source_files = [
    'kernels.c',
    'epanechnikov.c',
    'gaussian.c',
    'testkernel.c',
]
_cwd = Path(__file__).ancestor(1)

files = map(lambda x: _cwd.child(x), _source_files)
