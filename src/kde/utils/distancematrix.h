#ifndef KERNELS_DISTANCEMATRIX_H
#define KERNELS_DISTANCEMATRIX_H

#include <gsl/gsl_matrix.h>

void computeDistanceMatrix(gsl_matrix* patterns, gsl_matrix* distanceMatrix);

#endif //KERNELS_DISTANCEMATRIX_H
