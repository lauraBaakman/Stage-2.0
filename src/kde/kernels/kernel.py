class Kernel(object):
    def __init__(self):
        pass

    def evaluate(self, xs):
        return self._implementation.evaluate(xs)

    def scaling_factor(self):
        raise NotImplementedError()

    def to_C_enum(self):
        raise NotImplementedError()