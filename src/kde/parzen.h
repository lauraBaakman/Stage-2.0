#ifndef PARZEN_H
#define PARZEN_H

#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_matrix.h>

#include "kernels/kernels.h"

void parzen(gsl_matrix* xs, gsl_matrix* xis, double windowWidth, SymmetricKernel kernel,
            gsl_vector* outDensities);

#endif //PARZEN_H
