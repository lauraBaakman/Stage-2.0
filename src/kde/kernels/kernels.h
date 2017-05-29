#ifndef KERNELS_KERNELS_H
#define KERNELS_KERNELS_H

#include "../utils/gsl_utils.h"
#include <stdbool.h>

typedef void (*KernelFreeFunction)(void);

typedef void (*SymmetricKernelPrepareFunction)(size_t dimensionality);
typedef double (*SymmetricKernelDensityFunction)(gsl_vector* pattern);

typedef void (*ShapeAdaptiveKernelAllocFunction)(size_t dimensionality);
typedef void (*ShapeAdaptiveKernelConstantsFunction)(gsl_matrix* globalBandwidthMatrix);
typedef double (*ShapeAdaptiveKernelDensityFunction)(gsl_vector* pattern, double localBandwidth);

typedef struct SymmetricKernel {
    SymmetricKernelDensityFunction density;
    SymmetricKernelPrepareFunction prepare;
    KernelFreeFunction free;
} SymmetricKernel;

typedef struct ShapeAdaptiveKernel {
    ShapeAdaptiveKernelDensityFunction  density;
    ShapeAdaptiveKernelAllocFunction allocate;
    ShapeAdaptiveKernelConstantsFunction computeConstants;
    KernelFreeFunction free;
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
