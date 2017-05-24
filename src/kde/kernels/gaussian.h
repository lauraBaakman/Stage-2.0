#ifndef KERNELS_GAUSSIAN_H
#define KERNELS_GAUSSIAN_H

#include <math.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#include "kernels.h"

/* Symmetric */
extern Kernel standardGaussianKernel;

/* Shape Adaptive */
extern Kernel shapeAdaptiveGaussianKernel;

#endif //KERNELS_GAUSSIAN_H
