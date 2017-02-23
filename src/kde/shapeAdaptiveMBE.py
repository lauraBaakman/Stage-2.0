from kde.modifeidbreiman import ModifiedBreimanEstimator
import kde.utils.automaticWindowWidthMethods as automaticWindowWidthMethods
from kde.estimatorimplementation import EstimatorImplementation
from kde.parzen import ParzenEstimator


class ShapeAdaptiveModifiedBreimanEstimator(ModifiedBreimanEstimator):
    """
    Implementation of the shape adaptive modified Breiman Estimator.
    """

    def __init__(self, dimension, sensitivity=1 / 2,
                 pilot_window_width_method=automaticWindowWidthMethods.ferdosi,
                 number_of_grid_points=ModifiedBreimanEstimator.default_number_of_grid_points,
                 pilot_kernel=None, pilot_estimator_implementation=None,
                 kernel=None, final_estimator_implementation=None):
        super().__init__(dimension, kernel, sensitivity, pilot_kernel, pilot_window_width_method, number_of_grid_points,
                         pilot_estimator_implementation, final_estimator_implementation)
        self._pilot_estimator_implementation = pilot_estimator_implementation or ParzenEstimator
        self._final_estimator_implementation = final_estimator_implementation or _ShapeAdaptiveModifiedBreimanEstimator_C


class _ShapeAdaptiveModifiedBreimanEstimator(EstimatorImplementation):
    def __init__(self, xi_s, x_s, dimension, kernel, local_bandwidths, general_bandwidth):
        super().__init__(xi_s, x_s, dimension, kernel, general_bandwidth)
        self._local_bandwidths = local_bandwidths.astype(float, copy=False)


class _ShapeAdaptiveModifiedBreimanEstimator_C(_ShapeAdaptiveModifiedBreimanEstimator):

    def estimate(self):
        raise NotImplementedError("The C implementation of the Shape Adaptive Modified Breiman is estimator has not "
                              "yet been written.")


class _ShapeAdaptiveModifiedBreimanEstimator_Python(_ShapeAdaptiveModifiedBreimanEstimator):
    def estimate(self):
        raise NotImplementedError("The Python implementation of the Shape Adaptive Modified Breiman is estimator "
                                  "has not yet been written.")