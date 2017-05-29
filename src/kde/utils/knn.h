#ifndef KERNELS_KNN_H
#define KERNELS_KNN_H

#include <gsl/gsl_matrix.h>

void computeKNearestNeighbours(size_t k, size_t patternIdx,
                               gsl_matrix *patterns, gsl_matrix *distanceMatrix,
                               gsl_matrix *outNearestNeighbours);

#endif //KERNELS_KNN_H
