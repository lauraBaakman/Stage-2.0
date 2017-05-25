#include <gsl/gsl_vector_double.h>
#include "epanechnikov.ih"

Kernel epanechnikovKernel = {
        .isShapeAdaptive = false,
        .kernel.symmetricKernel.densityFunction = epanechnikovPDF,
        .kernel.symmetricKernel.factorFunction = epanechnikovConstant,
};

double epanechnikovConstant(int dimensionality) {
    double numerator = pow(M_PI, dimensionality / 2.0);
    double denominator = gamma(dimensionality / 2.0 + 1);
    return 2 * (numerator / denominator);
}

double epanechnikovPDF(gsl_vector* pattern, double constant) {
    double dotProduct = 0.0;
    gsl_blas_ddot(pattern,  pattern, &dotProduct);
    if (dotProduct >= 1) return 0;
    double numerator = (double) pattern->size + 2;
    return (numerator / constant) * (1 - pattern->size);
}
