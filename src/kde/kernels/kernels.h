#ifndef KERNELS_KERNELS_H
#define KERNELS_KERNELS_H

#include "../utils/gsl_utils.h"
#include <stdbool.h>

typedef double (*SymmetricKernelDensityFunction)(gsl_vector* pattern, double factor);
typedef double (*SymmetricKernelConstantFunction)(int dimensionality);

typedef double (*ShapeAdaptiveKernelDensityFunction)(gsl_vector* pattern, double localBandwidth,
                                                     double globalScalingFactor, gsl_matrix * globalInverse, double gaussianConstant,
                                                     gsl_vector* scaledPattern);
typedef void (*ShapeAdaptiveKernelConstantFunction)(gsl_matrix* globalBandwidthMatrix,
                                                           gsl_matrix* outGlobalInverse, double* outGlobalScalingFactor, double* outPDFConstant);

typedef struct SymmetricKernel {
    SymmetricKernelConstantFunction factorFunction;
    SymmetricKernelDensityFunction densityFunction;
} SymmetricKernel;

typedef struct ShapeAdaptiveKernel {
    ShapeAdaptiveKernelConstantFunction  factorFunction;
    ShapeAdaptiveKernelDensityFunction  densityFunction;
} ShapeAdaptiveKernel;

typedef union {
    SymmetricKernel symmetricKernel;
    ShapeAdaptiveKernel shapeAdaptiveKernel;
} kernelUnion;

typedef struct Kernel {
    bool isShapeAdaptive;
    kernelUnion kernel;
} Kernel;

typedef enum {
    TEST = 0,
    STANDARD_GAUSSIAN = 1,
    EPANECHNIKOV = 2,
    SHAPE_ADAPTIVE_EPANECHNIKOV = 3,
    SHAPE_ADAPTIVE_GAUSSIAN = 4,
} KernelType;

Kernel selectKernel(KernelType type);
SymmetricKernel selectSymmetricKernel(KernelType type);
ShapeAdaptiveKernel selectShapeAdaptiveKernel(KernelType type);

double computeScalingFactor(double generalBandwidth, gsl_matrix* covarianceMatrix);

double computeLocalScalingFactor(double globalScalingFactor, double localBandwidth, size_t dimension);

#endif //KERNELS_KERNELS_H
