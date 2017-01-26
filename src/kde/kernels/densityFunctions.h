//
// Created by Laura Baakman on 19/12/2016.
//

#ifndef KERNELS_DENSITYFUNCTIONS_H
#define KERNELS_DENSITYFUNCTIONS_H

#include <printf.h>
#include <math.h>
#include "../utils.h"

double standardGaussianFactor(int patternDimensionality);
double standardGaussian(double* pattern, int patternDimensionality, double factor);

double epanechnikovDenominator(int dimensionality);
double epanechnikov(double *data, int dimensionality, double denominator);

//Auxilaries that should be in a internal header....
double dotProduct(double *a, double *b, int length);


#endif //KERNELS_DENSITYFUNCTIONS_H
