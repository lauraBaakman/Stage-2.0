import timeit

import numpy as np

import kde.kernels.shapeadaptivegaussian as sa_gaussian


def test_1():
    dimension = 3
    num_patterns = 60000
    local_bandwidth = np.random.rand(num_patterns)
    patterns = np.random.rand(num_patterns, dimension)

    H = np.array([[2, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 2]])

    kernel = sa_gaussian.ShapeAdaptiveGaussian(H)
    kernel.evaluate(patterns, local_bandwidth)


# def test_2():
#     dimension = 5
#     window_width = 0.25
#
#     patterns = np.random.rand(5000, dimension)
#     data_points = np.random.rand(500, dimension)


if __name__ == '__main__':
    num_runs = 30
    print(timeit.timeit(
        "test_1()",
        setup="from __main__ import test_1",
        number=30)/num_runs
    )
    # print(timeit.timeit(
    #     "test_2()",
    #     setup="from __main__ import test_2",
    #     number=5000)
    # )
