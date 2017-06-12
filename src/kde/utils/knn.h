#ifndef KERNELS_KNN_H
#define KERNELS_KNN_H

#include <gsl/gsl_matrix.h>

void computeKNearestNeighbours(gsl_vector* pattern, size_t k, gsl_matrix *outNearestNeighbours);

void nn_prepare(gsl_matrix* xs);

void nn_free();

#endif //KERNELS_KNN_H
