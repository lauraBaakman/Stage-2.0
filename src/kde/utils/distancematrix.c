#include <gsl/gsl_matrix.h>
#include "distancematrix.ih"

//void computeDistanceMatrix(Array *patterns, Array *distanceMatrix){
//
//    double* a = patterns->data;
//    double* b;
//    double distance;
//
//    arraySetDiagonalToZero(distanceMatrix);
//
//    for(int i = 0;
//            i < patterns->length;
//            i++, a+= patterns->rowStride)
//    {
//        b = a + patterns->rowStride;
//        for(int j = i + 1;
//                j < patterns->length;
//                j++, b+= patterns->rowStride){
//            distance = squaredEuclidean(a, b, patterns->dimensionality);
//            arraySetElement(distanceMatrix, i, j, distance);
//            arraySetElement(distanceMatrix, j, i, distance);
//        }
//    }
//}

double squaredEuclidean(double* a, double* b, int length){
    double distance = 0;
    for(int i = 0; i < length; i++){
        distance += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return distance;
}

void computeDistanceMatrix(gsl_matrix *patterns, gsl_matrix *distanceMatrix) {
    gsl_matrix_set_identity(distanceMatrix);

    size_t num_patterns = patterns->size1;
    size_t dimension = patterns->size2;

    double distance;
    for(int i = 0; i < distanceMatrix->size1; i++){
        for(int j = i + 1; j < distanceMatrix->size1; j++){
//            distance = squaredEuclidean(a, b, dimension);
            distance = 42.0;
            gsl_matrix_set(distanceMatrix, i, j, distance);
            gsl_matrix_set(distanceMatrix, j, i, distance);
        }
    }
}
