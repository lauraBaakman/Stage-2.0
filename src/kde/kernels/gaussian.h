#ifndef KERNELS_GAUSSIAN_H
#define KERNELS_GAUSSIAN_H

#include "kernels.h"

/* Symmetric */
extern Kernel standardGaussianKernel;

double standardGaussianConstant(int patternDimensionality);
double standardGaussianPDF(double *pattern, int patternDimensionality, double constant);

/* Shape Adaptive */

#endif //KERNELS_GAUSSIAN_H
