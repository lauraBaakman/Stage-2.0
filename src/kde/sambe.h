#ifndef SAMBE_H
#define SAMBE_H

#include <gsl/gsl_matrix.h>

#include "utils.h"
#include "kernels/kernels.h"

double sambeFinalDensity(double *pattern, Array *datapoints,
                         double globalBandwidth,
                         ShapeAdaptiveKernel kernel);

#endif //SAMBE_H
