import timeit

import numpy as np


def test_1():
    from kde.kernels.standardGaussian_new import StandardGaussian_New as the_class
    gaussian = the_class(dimension=3)
    matrix = np.random.rand(10000, 3)
    gaussian.evaluate(matrix)


def test_2():
    from kde.kernels.standardGaussian import StandardGaussian as the_class
    gaussian = the_class(dimension=3)
    matrix = np.random.rand(10000, 3)
    gaussian.evaluate(matrix)

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