#ifndef KERNELS_COVARIANCEMATRIX_H
#define KERNELS_COVARIANCEMATRIX_H

#include <gsl/gsl_matrix.h>

void computeCovarianceMatrix(gsl_matrix* patterns, gsl_matrix* covarianceMatrix);

#endif //KERNELS_COVARIANCEMATRIX_H
