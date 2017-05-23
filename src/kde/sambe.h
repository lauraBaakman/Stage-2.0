#ifndef SAMBE_H
#define SAMBE_H

#include <gsl/gsl_matrix.h>

#include "kernels/kernels.h"

void sambeFinalDensity(gsl_matrix* xs,
                       gsl_vector* localBandwidths, double globalBandwidth,
                       ShapeAdaptiveKernel kernel,
                       gsl_vector* outDensities);

#endif //SAMBE_H