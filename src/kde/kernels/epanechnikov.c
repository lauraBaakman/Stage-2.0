#include "epanechnikov.ih"

Kernel epanechnikovKernel = {
        .isShapeAdaptive = false,
        .kernel.symmetricKernel.density = normal_pdf,
        .kernel.symmetricKernel.prepare = normal_prepare,
        .kernel.symmetricKernel.free = normal_free,
};

static double squareRootOfTheVariance = 0.8728715609439694;

static double g_normal_constant;
static double g_normal_one_over_unit_variance_constant;
static gsl_vector* g_scaledPattern;

void normal_prepare(size_t dimension) {
    g_normal_constant = normal_constant(dimension) * normal_unit_variance_constant(dimension);
    g_normal_one_over_unit_variance_constant = 1.0 / squareRootOfTheVariance;
    g_scaledPattern = gsl_vector_alloc(dimension);
}

void normal_free() {
    g_normal_constant = 0.0;
    g_normal_one_over_unit_variance_constant = 0.0;
    gsl_vector_free(g_scaledPattern);
}

double normal_pdf(gsl_vector *pattern) {
    gsl_vector_memcpy(g_scaledPattern, pattern);
    gsl_vector_scale(g_scaledPattern, g_normal_one_over_unit_variance_constant);

    double dotProduct = 0.0;
    gsl_blas_ddot(g_scaledPattern,  g_scaledPattern, &dotProduct);

    if (dotProduct >= 1) return 0;

    return g_normal_constant * (1 - dotProduct);
}

double normal_constant(size_t dimension){
    return ((double) (dimension + 2)) / (2 * unitSphereVolume(dimension));
}

double unitSphereVolume(size_t dimension) {
    double numerator = pow(M_PI, dimension / 2.0);
    double denominator = gamma(dimension / 2.0 + 1);
    return numerator / denominator;
}

double normal_unit_variance_constant(size_t dimension) {
    return pow(1.0 / squareRootOfTheVariance, dimension);
}
