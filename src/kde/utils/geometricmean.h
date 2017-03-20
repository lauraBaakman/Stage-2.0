#ifndef KERNELS_GEOMETRICMEAN_H
#define KERNELS_GEOMETRICMEAN_H

#include <stddef.h>

double computeGeometricMean(double *values, size_t length);
double gsl_geometric_mean(const gsl_vector *vector);

#endif //KERNELS_GEOMETRICMEAN_H
