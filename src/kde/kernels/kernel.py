class Kernel(object):
    def __init__(self):
        pass

    def evaluate(self, xs):
        return self._implementation.evaluate(xs)

    def to_C_enum(self):
        raise NotImplementedError()
