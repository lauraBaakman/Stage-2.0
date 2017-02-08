//
// Created by Laura Baakman on 19/12/2016.
//

#ifndef KERNELS_DENSITYFUNCTIONS_H
#define KERNELS_DENSITYFUNCTIONS_H

#include <printf.h>
#include <math.h>
#include "../utils.h"

typedef double (*KernelDensityFunction)(double* data, int dimensionality, double factor);
typedef double (*KernelConstantFunction)(int dimensionality);
typedef struct Kernel {
    KernelConstantFunction factorFunction;
    KernelDensityFunction densityFunction;
} Kernel;

double standardGaussianConstant(int patternDimensionality);
double standardGaussianPDF(double *pattern, int patternDimensionality, double constant);

double epanechnikovConstant(int dimensionality);
double epanechnikovPDF(double *data, int dimensionality, double constant);

double testKernelConstant(int patternDimensionality);
double testKernelPDF(double *data, int dimensionality, double constant);

extern Kernel standardGaussianKernel;
extern Kernel epanechnikovKernel;
extern Kernel testKernel;

//Auxilaries that should be in a internal header....
double dotProduct(double *a, double *b, int length);

#endif //KERNELS_DENSITYFUNCTIONS_H
