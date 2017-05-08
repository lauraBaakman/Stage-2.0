class Component(object):

    def __init__(self):
        super(Component, self).__init__()

    def patterns(self, num_patterns):
        raise NotImplementedError()

    def densities(self, patterns):
        raise NotImplementedError()

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)