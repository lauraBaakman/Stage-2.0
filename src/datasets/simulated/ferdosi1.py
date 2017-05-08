import scipy.stats as stats
import numpy as np

from inputoutput.dataset import DataSet


def ferdosi_1(num_patterns=60000):
    patterns = _compute_patterns(num_patterns)
    densities = _compute_densities(patterns)
    data_set = DataSet(patterns=patterns,
                       densities=densities)
    return data_set


def _compute_patterns(num_patterns):
    trivariate_gaussian = _trivariate_gaussian(round(2/3.0 * num_patterns))
    background = _uniform_random_background(round(1/3.0 * num_patterns))

    return np.vstack((
        trivariate_gaussian,
        background
    ))


def _trivariate_gaussian(num_patterns):
    mean = np.array([50, 50, 50])
    covariance = np.diag(np.array([30, 30, 30]))

    patterns = np.random.multivariate_normal(mean, covariance, num_patterns)
    return patterns


def _uniform_random_background(num_patterns):
    x = np.random.uniform(0, 100, num_patterns)
    y = np.random.uniform(0, 100, num_patterns)
    z = np.random.uniform(0, 100, num_patterns)

    patterns = np.stack((x, y, z), 1)
    return patterns


def _compute_densities(patterns):
    trivariate_gaussian_densities = _compute_trivariate_gaussian_densities(patterns)
    uniform_densities = _compute_uniform_densities(patterns)

    return (trivariate_gaussian_densities + uniform_densities) / 2.0


def _compute_trivariate_gaussian_densities(patterns):
    mean = np.array([50, 50, 50])
    covariance = np.diag(np.array([30, 30, 30]))
    densities = stats.multivariate_normal.pdf(x=patterns,
                                              mean=mean,
                                              cov=covariance)
    return densities


def _compute_uniform_densities(patterns):
    densities_1D = stats.uniform.pdf(patterns, loc=0, scale=100)
    densities = np.prod(densities_1D, axis=1)
    return densities