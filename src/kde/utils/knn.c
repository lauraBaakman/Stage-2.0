#include "knn.ih"

Array* compute_k_nearest_neighbours(int k, int patterndIdx, Array *patterns, Array *distanceMatrix,
                                    Array *nearestNeighbours){
    double* distances = arrayGetRow(distanceMatrix, patterndIdx);
    return nearestNeighbours;
}
