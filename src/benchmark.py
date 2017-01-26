import timeit

import numpy as np


def test_1():
    dimension = 5
    window_width = 0.25

    import kde._kde as kde
    patterns = np.random.rand(5000, dimension)
    data_points = np.random.rand(500, dimension)
    (num_patterns, dimension) = patterns.shape
    densities = np.empty(num_patterns, dtype=float)
    kde.parzen_standard_gaussian(patterns, data_points, window_width, densities)


def test_2():
    dimension = 5
    window_width = 0.25

    import kde
    patterns = np.random.rand(5000, dimension)
    data_points = np.random.rand(500, dimension)
    (num_patterns, dimension) = patterns.shape

    kernel_shape = window_width * window_width * dimension
    kernel = kde.kernels.Gaussian(covariance_matrix=kernel_shape)

    estimator = kde.Parzen(dimension, window_width, kernel)
    densities = estimator.estimate(xi_s=data_points, x_s=patterns)

if __name__ == '__main__':
    print(timeit.timeit(
        "test_1()",
        setup="from __main__ import test_1",
        number=5000)
    )
    print(timeit.timeit(
        "test_2()",
        setup="from __main__ import test_2",
        number=5000)
    )