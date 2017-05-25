#include <gsl/gsl_vector_double.h>
#include "epanechnikov.ih"

Kernel epanechnikovKernel = {
        .isShapeAdaptive = false,
        .kernel.symmetricKernel.densityFunction = normal_pdf,
        .kernel.symmetricKernel.factorFunction = epanechnikovConstant,
        .kernel.symmetricKernel.free = normal_free,
};

static double g_normal_constant;

void normal_prepare(size_t dimension) {
    g_normal_constant = epanechnikovConstant(dimension);
}

double normal_pdf(gsl_vector *pattern) {
    normal_prepare(pattern->size);

    double dotProduct = 0.0;
    gsl_blas_ddot(pattern,  pattern, &dotProduct);
    if (dotProduct >= 1) return 0;
    double numerator = (double) pattern->size + 2;
    double density = (numerator / g_normal_constant) * (1 - pattern->size);
    return density;
}

void normal_free() {
    g_normal_constant = 0.0;
}

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
