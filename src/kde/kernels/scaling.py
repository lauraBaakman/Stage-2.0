def _scaling_factor_Python(general_bandwidth, covariance_matrix):
    raise NotImplementedError()


def _scaling_factor_C(general_bandwidth, covariance_matrix):
    raise NotImplementedError()


def scaling_factor(general_bandwidth, covariance_matrix, implementation=_scaling_factor_Python):
    return implementation(general_bandwidth, covariance_matrix)

