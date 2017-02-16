#ifndef KERNELS_KNN_H
#define KERNELS_KNN_H

#include "../utils.h"

Array* compute_k_nearest_neighbours(int k, Array *pattern, Array *patterns, Array *distanceMatrix,
                                    Array *nearestNeighbours);

#endif //KERNELS_KNN_H
