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

