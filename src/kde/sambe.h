#ifndef SAMBE_H
#define SAMBE_H

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector_double.h>

#include "kernels/kernels.h"

void sambe(gsl_matrix *xs,
           gsl_vector *localBandwidths, double globalBandwidth,
           KernelType kernel, int k,
           gsl_vector *outDensities);

#endif //SAMBE_H