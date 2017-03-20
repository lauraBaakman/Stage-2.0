#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include "../utils.h"

#ifndef KERNELS_EIGENVALUES_H
#define KERNELS_EIGENVALUES_H

#include <gsl/gsl_matrix.h>

void computeEigenValues(Array* data, Array* eigenValues);
gsl_vector* computeEigenValues2(gsl_matrix* matrix);


#endif //KERNELS_EIGENVALUES_H
