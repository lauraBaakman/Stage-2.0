#ifndef KERNELS_KERNELS_H
#define KERNELS_KERNELS_H

#include <gsl/gsl_matrix.h>
#include <stdbool.h>
#include "../utils.h"

typedef double (*SymmetricKernelDensityFunction)(double* data, int dimensionality, double factor);
typedef double (*SymmetricKernelConstantFunction)(int dimensionality);

typedef double (*ASymmetricKernelDensityFunction)(gsl_vector* pattern, gsl_vector* mean, gsl_matrix* shapeMatrix);
typedef gsl_matrix* (*ASymmetricKernelConstantFunction)(Array* covarianceMatrix);

typedef double (*ShapeAdaptiveKernelDensityFunction)(gsl_vector* pattern, double localBandwidth, gsl_matrix* globalBandwidthMatrix,
                                                     double globalScalingFactor, gsl_matrix * globalInverse);
typedef gsl_matrix* (*ShapeAdaptiveKernelConstantFunction)(Array* globalBandwidthMatrix,
                                                           gsl_matrix* outGlobalInverse, double* outGlobalScalingFactor);

typedef struct SymmetricKernel {
    SymmetricKernelConstantFunction factorFunction;
    SymmetricKernelDensityFunction densityFunction;
} SymmetricKernel;

typedef struct ASymmetricKernel {
    ASymmetricKernelConstantFunction factorFunction;
    ASymmetricKernelDensityFunction densityFunction;
} ASymmetricKernel;

typedef struct ShapeAdaptiveKernel {
    ShapeAdaptiveKernelConstantFunction  factorFunction;
    ShapeAdaptiveKernelDensityFunction  densityFunction;
} ShapeAdaptiveKernel;

typedef union {
    SymmetricKernel symmetricKernel;
    ASymmetricKernel aSymmetricKernel;
    ShapeAdaptiveKernel shapeAdaptiveKernel;
} kernelUnion;

typedef struct Kernel {
    bool isSymmetric;
    bool isShapeAdaptive;
    kernelUnion kernel;
} Kernel;

typedef enum {
    TEST = 0,
    STANDARD_GAUSSIAN = 1,
    EPANECHNIKOV = 2,
    GAUSSIAN = 3,
    SHAPE_ADAPTIVE_GAUSSIAN = 4,
} KernelType;

Kernel selectKernel(KernelType type);
SymmetricKernel selectSymmetricKernel(KernelType type);
ASymmetricKernel selectASymmetricKernel(KernelType type);
ShapeAdaptiveKernel selectShapeAdaptiveKernel(KernelType type);

double computeScalingFactor(double generalBandwidth, gsl_matrix_view covarianceMatrix);

extern Kernel standardGaussianKernel;
extern Kernel epanechnikovKernel;
extern Kernel testKernel;
extern Kernel gaussianKernel;
extern Kernel shapeAdaptiveGaussianKernel;

#endif //KERNELS_KERNELS_H
