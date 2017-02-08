#ifndef KERNELS_KERNELS_H
#define KERNELS_KERNELS_H

typedef double (*KernelDensityFunction)(double* data, int dimensionality, double factor);
typedef double (*KernelConstantFunction)(int dimensionality);
typedef struct Kernel {
    KernelConstantFunction factorFunction;
    KernelDensityFunction densityFunction;
} Kernel;

typedef enum {
    STANDARDGAUSSIAN,
    EPANECHNIKOV,
    TEST,
} KernelType;

Kernel selectKernel(KernelType type);

extern Kernel standardGaussianKernel;
extern Kernel epanechnikovKernel;
extern Kernel testKernel;

#endif //KERNELS_KERNELS_H
