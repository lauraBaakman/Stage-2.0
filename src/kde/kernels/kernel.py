class Kernel(object):
    def __init__(self):
        pass

    def evaluate(self, xs):
        raise NotImplementedError()

    def to_C_enum(self):
        raise NotImplementedError()

    def _get_data_dimension(self, xs):
        if xs.ndim == 1:
            (dimension,) = xs.shape
        elif xs.ndim == 2:
            (_, dimension) = xs.shape
        else:
            raise TypeError("Expected a vector or a matrix, not a {}-dimensional array.".format(xs.ndim))
        return dimension


class KernelException(Exception):

    def __init__(self, message, *args):
        super(KernelException, self).__init__(message, *args)