import math

import scipy as sp


def _volume_nd_unit_sphere(dimension):
    numerator = math.pow(math.pi, dimension / 2)
    denominator = sp.special.gamma((dimension / 2) + 1)
    return numerator / denominator


def silverman(data_points):
    (N, dimension) = data_points.shape
    term = 8 * (dimension + 4) * math.pow(2 * math.sqrt(math.pi), dimension) / _volume_nd_unit_sphere(dimension)
    average_variance = sp.mean(sp.var(data_points, axis=0))
    return math.pow(term, 1 / (dimension + 4)) * math.pow(N, -1 / (dimension + 4)) * average_variance


def ferdosi(datapoints):
    raise NotImplementedError()