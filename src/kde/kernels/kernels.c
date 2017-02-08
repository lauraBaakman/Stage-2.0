//
// Created by Laura Baakman on 19/12/2016.
//

#include "kernels.ih"

Kernel standardGaussianKernel = {
        .factorFunction = standardGaussianConstant,
        .densityFunction = standardGaussianPDF,
};

Kernel epanechnikovKernel = {
        .factorFunction = epanechnikovConstant,
        .densityFunction = epanechnikovPDF,
};

Kernel testKernel = {
        .factorFunction = testKernelConstant,
        .densityFunction = testKernelPDF,
};


Kernel selectKernel(KernelType type){
    switch (type) {
        case EPANECHNIKOV:
            return epanechnikovKernel;
        case STANDARDGAUSSIAN:
            return standardGaussianKernel;
        case TEST:
            return testKernel;
        default:
            fprintf(stderr, "%d is an invalid kernel type.\n", type);
            exit(-1);
    }
}

double standardGaussianConstant(int patternDimensionality){
    return pow(2 * M_PI, - 1 * patternDimensionality * 0.5);
}

double standardGaussianPDF(double *pattern, int patternDimensionality, double constant){
    double dotProduct = 0.0;
    for(int i = 0; i < patternDimensionality; i++) {
        dotProduct += pattern[i] * pattern[i];
    }
    return constant * exp(-0.5 * dotProduct);
}

double epanechnikovConstant(int dimensionality){
    double numerator = pow(M_PI, dimensionality / 2.0);
    double denominator = gamma(dimensionality / 2.0 + 1);
    return 2 * (numerator / denominator);
}

double epanechnikovPDF(double *data, int dimensionality, double constant) {
    double patternDotPattern = dotProduct(data, data, dimensionality);
    if (patternDotPattern >= 1) {
        return 0;
    }
    double numerator = (double) dimensionality + 2;
    return (numerator / constant) * (1 - patternDotPattern);
}

double testKernelConstant(int patternDimensionality){
    return 1.0 / patternDimensionality;
}

double testKernelPDF(double *data, int dimensionality, double constant){
    double density = 0;
    for (int i = 0; i < dimensionality; i++){
        density += data[i];
    }
    double mean = density * constant;
    return fabs(mean);
}

double dotProduct(double *a, double *b, int length) {
    double dotProduct = 0;
    for (int i = 0; i < length; ++i) {
        dotProduct += (a[i] * b[i]);
    }
    return dotProduct;
}