#ifndef KERNELS_MODIFEID_BREIMAN_H
#define KERNELS_MODIFEID_BREIMAN_H

#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_matrix.h>

#include "kernels/kernels.h"

void mbe(gsl_matrix* xs, gsl_matrix *xis,
         double globalBandwidth, gsl_vector *localBandwidths,
         KernelType kernelType,
         gsl_vector *densities);

#endif //KERNELS_MODIFEID_BREIMAN_H
