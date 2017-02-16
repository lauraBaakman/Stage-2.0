def compute_distance_matrix(patterns, implementation=None):
    actual_implementation = implementation or _compute_distance_matrix_C
    return actual_implementation(patterns)


def _compute_distance_matrix_Python(patterns):
    raise NotImplementedError()


def _compute_distance_matrix_C(patterns):
    raise NotImplementedError()