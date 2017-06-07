import scipy.spatial as spatial


def compute_distance_matrix(patterns):
    distance_euclidean = spatial.distance_matrix(patterns, patterns, p=2)
    distance_squared_euclidean = distance_euclidean ** 2
    return distance_squared_euclidean
