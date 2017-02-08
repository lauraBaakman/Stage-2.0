class DataSet(object):
    def __init__(self, patterns):
        self._patterns = patterns
        self._results = None

    @property
    def num_patterns(self):
        raise NotImplementedError()

    @property
    def dimension(self):
        raise NotImplementedError()

    @property
    def patterns(self):
        raise NotImplementedError()

    @property
    def results(self):
        if not self._results:
            raise LookupError("Results has not yet been set.")
        return self._results

    @classmethod
    def from_file(cls, file):
        return _DataSetReader(file).read()

    def result_to_file(self, file):
        _ResultsWriter(file).write()


class _DataSetReader(object):
    def __init__(self, file):
        self.in_file = file

    def read(self):
        raise NotImplementedError()


class _ResultsWriter(object):
    def __init__(self, file):
        self.out_file = file

    def write(self):
        raise NotImplementedError()
