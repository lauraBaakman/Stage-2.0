#ifndef KERNELS_KNN_H
#define KERNELS_KNN_H

#include "../utils.h"

void compute_k_nearest_neighbours(int k, int patternIdx,
                                    gsl_matrix* patterns, gsl_matrix *distanceMatrix,
                                    gsl_matrix *outNearestNeighbours);

#endif //KERNELS_KNN_H
