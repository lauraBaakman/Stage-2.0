#ifndef KERNELS_KERNELS_H
#define KERNELS_KERNELS_H

typedef double (*KernelDensityFunction)(double* data, int dimensionality, double factor);
typedef double (*KernelConstantFunction)(int dimensionality);
typedef struct Kernel {
    KernelConstantFunction factorFunction;
    KernelDensityFunction densityFunction;
} Kernel;

typedef enum {
    TEST = 0,
    STANDARD_GAUSSIAN = 1,
    EPANECHNIKOV = 2,
} KernelType;

Kernel selectKernel(KernelType type);

#endif //KERNELS_KERNELS_H
