#ifndef KERNELS_DENSITYFUNCTIONS_H
#define KERNELS_DENSITYFUNCTIONS_H

typedef double (*KernelDensityFunction)(double* data, int dimensionality, double factor);
typedef double (*KernelConstantFunction)(int dimensionality);
typedef struct Kernel {
    KernelConstantFunction factorFunction;
    KernelDensityFunction densityFunction;
} Kernel;

extern Kernel standardGaussianKernel;
extern Kernel epanechnikovKernel;
extern Kernel testKernel;

#endif //KERNELS_DENSITYFUNCTIONS_H
