import timeit

import numpy as np

from kde.sambe import SAMBEstimator


def test_1():
    dimension = 3
    num_patterns = 250
    x_s = np.random.rand(num_patterns, dimension)
    xi_s = np.random.rand(num_patterns, dimension)
    pilot_densities = np.random.rand(num_patterns, dimension)
    general_bandwidth = 0.7
    estimator = SAMBEstimator(dimension=dimension)
    estimator.estimate(
        x_s=x_s, xi_s=xi_s,
        pilot_densities=pilot_densities,
        general_bandwidth=general_bandwidth
    )

# def test_2():
#     dimension = 5
#     window_width = 0.25
#
#     patterns = np.random.rand(5000, dimension)
#     data_points = np.random.rand(500, dimension)


if __name__ == '__main__':
    num_runs = 5
    total_time = timeit.timeit(
        "test_1()",
        setup="from __main__ import test_1",
        number=30)
    print(total_time / num_runs)
    # print(timeit.timeit(
    #     "test_2()",
    #     setup="from __main__ import test_2",
    #     number=5000)
    # )
