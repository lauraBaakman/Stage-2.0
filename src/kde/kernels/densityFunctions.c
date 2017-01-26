//
// Created by Laura Baakman on 19/12/2016.
//

#include "densityFunctions.h"

double standardGaussianFactor(int patternDimensionality){
    return pow(2 * M_PI, - 1 * patternDimensionality * 0.5);
}

double standardGaussian(double* pattern, int patternDimensionality, double factor){
    double dotProduct = 0.0;
    for(int i = 0; i < patternDimensionality; i++) {
        dotProduct += pattern[i] * pattern[i];
    }
    return factor * exp(-0.5 * dotProduct);
}

double epanechnikovDenominator(int dimensionality){
    double numerator = pow(M_PI, dimensionality / 2.0);
    double denominator = gamma(dimensionality / 2.0 + 1);
    return 2 * (numerator / denominator);
}

double epanechnikov(double *data, int dimensionality, double denominator) {
    double patternDotPattern = dotProduct(data, data, dimensionality);
    if (patternDotPattern >= 1) {
        return 0;
    }
    double numerator = (double) dimensionality + 2;
    return (numerator / denominator) * (1 - patternDotPattern);
}

double dotProduct(double *a, double *b, int length) {
    double dotProduct = 0;
    for (int i = 0; i < length; ++i) {
        dotProduct += (a[i] * b[i]);
    }
    return dotProduct;
}