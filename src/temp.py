import kde.kernels._kernels as _kernels
import numpy as np

if __name__ == '__main__':
    data = np.array([[1.0, 2.0]])

    density = _kernels.epanechnikov_single_pattern(data)
    print(density)
