#include "gaussian.ih"
#include <math.h>

Kernel standardGaussianKernel = {
        .isShapeAdaptive = false,
        .kernel.symmetricKernel.densityFunction = standardGaussianPDF,
        .kernel.symmetricKernel.factorFunction = standardGaussianConstant,
};

double standardGaussianConstant(int patternDimensionality){
    return pow(2 * M_PI, -1 * patternDimensionality * 0.5);
}

double standardGaussianPDF(double *pattern, int patternDimensionality, double constant){
    double dotProduct = 0.0;
    for ( int i = 0; i < patternDimensionality; i++ ) {
        dotProduct += pattern[i] * pattern[i];
    }
    return constant * exp(-0.5 * dotProduct);
}