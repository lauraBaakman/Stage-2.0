import kde
import kde.windowWidthMethods
import kde.kernels as kernels


class ModifiedBreimanEstimator(object):

    def __init__(self, dimension, kernel, sensitivity=1/2,
                 pilot_kernel=kernels.Epanechnikov(),
                 pilot_window_width_method=kde.windowWidthMethods.ferdosi()):
        self._pilot_kernel = pilot_kernel
        self._kernel = kernel
        self._dimension = dimension
        self._pilot_window_width_method = pilot_window_width_method
        self._sensitivity = sensitivity

    def estimate(self, xi_s, x_s=None):
        if x_s is None:
            x_s = xi_s

        # Compute pilot window width
        window_width = self._pilot_window_width_method(xi_s)

        # Compute grid for pilot densities

        # Compute pilot densities
        pilot_densities = kde.Parzen(
            window_width=window_width,
            dimension=self._dimension,
            kernel=self._pilot_kernel
        ).estimate(xi_s=xi_s, x_s=x_s)

        # Compute local bandwidths
        local_bandwidths = self._compute_local_bandwidths(pilot_densities)

        # Compute densities
        estimator =_MBEstimator(xi_s=xi_s, x_s=x_s,
                                dimension=self._dimension,
                                kernel=self._kernel, local_bandwidths=local_bandwidths)
        densities = estimator.estimate()

    def _compute_local_bandwidths(self, pilot_densities):
        raise NotImplementedError

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)


class _MBEstimator:

    def __init__(self, xi_s, x_s, dimension, kernel, local_bandwidths):
        self._xi_s = xi_s
        self._x_s = x_s
        self._dimension = dimension
        self._kernel = kernel
        self._local_bandwidths = local_bandwidths

    def estimate(self):
        raise NotImplementedError()
