class Kernel(object):
    def __init__(self):
        pass

    def evaluate(self, pattern):
        raise NotImplementedError()

    def to_C_enum(self):
        raise NotImplementedError()
