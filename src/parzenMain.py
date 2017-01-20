import numpy as np

import kde._kde as _kde
import kde

if __name__ == '__main__':
    patterns = np.array([[0.5, 0.5], [0.5, 1.5], [0.5, 0.7]])
    datapoints = np.array([[0.4, 0.4], [0.4, 1.4], [0.4, 0.6], [0.3, 0.4]])
    window_width = 0.25

    (num_patterns, dimension) = patterns.shape
    densities = np.empty(num_patterns, dtype=float)
    _kde.parzen_standard_gaussian(patterns, datapoints, window_width, densities)

    print(densities)

    # Python implementation
    kernel_shape = window_width * window_width * dimension
    kernel = kde.kernels.Gaussian(covariance_matrix=kernel_shape)
    estimator = kde.Parzen(dimension, window_width, kernel)
    densities = estimator.estimate(xi_s=datapoints, x_s=patterns)
    print(densities)



