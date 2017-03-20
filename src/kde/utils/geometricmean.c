#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_double.h>
#include "geometricmean.h"
#include "math.h"

double computeGeometricMean(double *values, size_t length) {
    double product = 0;
    for(size_t i = 0; i < length; i++){
        //Use log to avoid multiplying a lot of small numbers
        product += log(values[i]);
    }
    return exp(product / length);
}

double gsl_geometric_mean(const gsl_vector *vector) {
    return computeGeometricMean(vector->data, vector->size);
}
