#ifndef SAMBE_H
#define SAMBE_H

#include <gsl/gsl_matrix.h>

#include "utils.h"
#include "kernels/kernels.h"

double sambeFinalDensity(gsl_vector *pattern, gsl_matrix *datapoints,
                         double globalBandwidth,
                         ShapeAdaptiveKernel kernel);

#endif //SAMBE_H