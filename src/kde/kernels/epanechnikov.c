#include <gsl/gsl_linalg.h>
#include "epanechnikov.ih"

Kernel epanechnikovKernel = {
        .isShapeAdaptive = false,
        .kernel.symmetricKernel.density = normal_pdf,
        .kernel.symmetricKernel.prepare = normal_prepare,
        .kernel.symmetricKernel.free = normal_free,
};

Kernel shapeAdaptiveEpanechnikovKernel = {
        .isShapeAdaptive = true,
        .kernel.shapeAdaptiveKernel.density = sa_pdf,
        .kernel.shapeAdaptiveKernel.allocate = sa_allocate,
        .kernel.shapeAdaptiveKernel.free = sa_free,
        .kernel.shapeAdaptiveKernel.computeConstants = sa_computeConstants,
};

static double squareRootOfTheVariance = 0.8728715609439694;

static double g_normal_constant;
static double g_normal_one_over_unit_variance_constant;
static gsl_vector* g_scaledPattern;

static gsl_matrix* g_sa_globalInverse;
static gsl_matrix* g_sa_LUDecompositionH;
static gsl_permutation* g_sa_permutation;

double unitSphereVolume(size_t dimension) {
    double numerator = pow(M_PI, dimension / 2.0);
    double denominator = tgamma(dimension / 2.0 + 1);
    return numerator / denominator;
}

/* Normal Kernel */

void normal_prepare(size_t dimension) {
    g_normal_constant = normal_constant(dimension) * normal_unitVarianceConstant(dimension);
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

double normal_unitVarianceConstant(size_t dimension) {
    return pow(1.0 / squareRootOfTheVariance, dimension);
}

/* Shape Adaptive Kernel */

double sa_pdf(gsl_vector* pattern, double localBandwidth){
    printf("sa_pdf\n");
    return 42.0;
}

void sa_allocate(size_t dimension){
    g_sa_globalInverse = gsl_matrix_alloc(dimension, dimension);
    g_sa_LUDecompositionH = gsl_matrix_alloc(dimension, dimension);
    g_sa_permutation = gsl_permutation_alloc(dimension);

    sa_computeDimensionDependentConstants(dimension);
}

void sa_computeConstants(gsl_matrix *globalBandwidthMatrix){
    //Copy the global bandwidth matrix so that we can change it
    gsl_matrix_memcpy(g_sa_LUDecompositionH, globalBandwidthMatrix);

    //Compute LU decompostion
    int signum = 0;
    gsl_linalg_LU_decomp(g_sa_LUDecompositionH, g_sa_permutation, &signum);

    //Compute global inverse
    gsl_linalg_LU_invert(g_sa_LUDecompositionH, g_sa_permutation, g_sa_globalInverse);
}

void sa_computeDimensionDependentConstants(size_t dimension){
    printf("sa_computeDimensionDependentConstants\n");
}

void sa_free(){
    gsl_matrix_free(g_sa_globalInverse);
    gsl_matrix_free(g_sa_LUDecompositionH);
    gsl_permutation_free(g_sa_permutation);
}