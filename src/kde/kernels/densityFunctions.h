//
// Created by Laura Baakman on 19/12/2016.
//

#ifndef KERNELS_DENSITYFUNCTIONS_H
#define KERNELS_DENSITYFUNCTIONS_H

#include <printf.h>
#include <math.h>

double standardGaussianFactor(int patternDimensionality);
double standardGaussian(double* pattern, int patternDimensionality, double factor);

#endif //KERNELS_DENSITYFUNCTIONS_H
