#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector_double.h>
#include "distancematrix.ih"

void computeDistanceMatrix(gsl_matrix *patterns, gsl_matrix *distanceMatrix) {
    gsl_matrix_set_zero(distanceMatrix);

    gsl_vector_view a, b;

    size_t patternCount = patterns->size1;

    double distance;
    for(size_t i = 0; i < patternCount; i++){
        a = gsl_matrix_row(patterns, i);
        for(size_t j = i + 1; j < patternCount; j++){
            b = gsl_matrix_row(patterns, j);

            distance = squaredEuclidean(&a.vector, &b.vector);

            gsl_matrix_set(distanceMatrix, i, j, distance);
            gsl_matrix_set(distanceMatrix, j, i, distance);
        }
    }
}

double squaredEuclidean(gsl_vector* a, gsl_vector* b){
    size_t vectorDimension = a->size;
    double distance = 0;
    double difference;
    for(size_t i = 0; i < vectorDimension; i++){
        difference = a->data[i] - b->data[i];
        distance += difference * difference;
    }
    return distance;
}