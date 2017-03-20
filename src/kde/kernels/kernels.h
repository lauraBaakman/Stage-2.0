#ifndef KERNELS_KERNELS_H
#define KERNELS_KERNELS_H

#include <gsl/gsl_matrix.h>
#include <stdbool.h>
#include "../utils.h"

typedef double (*SymmetricKernelDensityFunction)(double* data, int dimensionality, double factor);
typedef double (*SymmetricKernelConstantFunction)(int dimensionality);

typedef double (*ASymmetricKernelDensityFunction)(gsl_vector* pattern, gsl_vector* mean, gsl_matrix* shapeMatrix);
typedef gsl_matrix* (*ASymmetricKernelConstantFunction)(Array* covarianceMatrix);

typedef struct SymmetricKernel {
    SymmetricKernelConstantFunction factorFunction;
    SymmetricKernelDensityFunction densityFunction;
} SymmetricKernel;

typedef struct ASymmetricKernel {
    ASymmetricKernelConstantFunction factorFunction;
    ASymmetricKernelDensityFunction densityFunction;
} ASymmetricKernel;

typedef union {
    SymmetricKernel symmetricKernel;
    ASymmetricKernel aSymmetricKernel;
} kernelUnion;

typedef struct Kernel {
    bool isSymmetric;
    kernelUnion kernel;
} Kernel;

typedef enum {
    TEST = 0,
    STANDARD_GAUSSIAN = 1,
    EPANECHNIKOV = 2,
    GAUSSIAN = 3,
} KernelType;

Kernel selectKernel(KernelType type);
SymmetricKernel selectSymmetricKernel(KernelType type);
ASymmetricKernel selectASymmetricKernel(KernelType type);

extern Kernel standardGaussianKernel;
extern Kernel epanechnikovKernel;
extern Kernel testKernel;
extern Kernel gaussianKernel;

#endif //KERNELS_KERNELS_H
