//
// Created by Laura Baakman on 19/12/2016.
//

#include "densityFunctions.h"

double standard_gaussian(double* pattern, int pattern_dimensionality){
    double factor = pow(2 * M_PI, - 1 * pattern_dimensionality * 0.5);
    double dot_product = 0.0;
    for(int i = 0; i < pattern_dimensionality; i++) {
        dot_product += pattern[i] * pattern[i];
    }
    return factor * exp(-0.5 * dot_product);
}