class Kernel(object):
    def __init__(self):
        pass

    def evaluate(self, xs):
        return self._implementation.evaluate(xs)

    def scaling_factor(self, general_bandwidth, eigen_values):
        raise NotImplementedError()

    def to_C_enum(self):
        raise NotImplementedError()


class KernelException(Exception):

    def __init__(self, message, *args):
        super(KernelException, self).__init__(message, *args)