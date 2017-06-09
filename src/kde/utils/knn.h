#ifndef KERNELS_KNN_H
#define KERNELS_KNN_H

#include <gsl/gsl_matrix.h>

void computeKNearestNeighboursOld(size_t k, size_t patternIdx, gsl_matrix *patterns,
                                  gsl_matrix *outNearestNeighbours);

void computeKNearestNeighbours(gsl_vector* pattern, size_t k, gsl_matrix *outNearestNeighbours);

void nn_prepare(gsl_matrix* xs);

void nn_free();

#endif //KERNELS_KNN_H
