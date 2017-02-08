import timeit

import numpy as np


def test_1():
    dimension = 5
    window_width = 0.25

    patterns = np.random.rand(5000, dimension)
    data_points = np.random.rand(500, dimension)

def test_2():
    dimension = 5
    window_width = 0.25

    patterns = np.random.rand(5000, dimension)
    data_points = np.random.rand(500, dimension)


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