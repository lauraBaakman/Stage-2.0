import numpy as np

import kde._kde as _kde

if __name__ == '__main__':
    patterns = np.array([[0.5, 0.5], [0.5, 1.5], [0.5, 0.7]])
    (num_patterns, _) = patterns.shape
    densities = np.empty(num_patterns, dtype=float)
    _kde.parzen_multi_pattern(patterns, densities)

    print(densities)
