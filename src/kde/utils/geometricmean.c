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
