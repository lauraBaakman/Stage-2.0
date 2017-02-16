def compute_distance_matrix(patterns, implementation=None):
    actual_implementation = implementation or compute_distance_matrix_C
    return actual_implementation(patterns)

def compute_distance_matrix_python(patterns):
    raise NotImplementedError()

def compute_distance_matrix_C(patterns):
    raise NotImplementedError()