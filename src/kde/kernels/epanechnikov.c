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

static int g_numThreads;

static gsl_vector** g_scaledPattern;

static double g_normal_constant;
static double g_normal_one_over_unit_variance_constant;

static gsl_vector* g_sa_scaledPattern;
static gsl_matrix* g_sa_globalInverse;
static gsl_matrix* g_sa_LUDecompositionH;
static gsl_permutation* g_sa_permutation;
static double g_sa_globalScalingFactor;
static double g_sa_epanechnikovConstant;


double epanechnikov_constant(size_t dimension){
    return ((double) (dimension + 2)) / (2 * unitSphereVolume(dimension));
}

double unitSphereVolume(size_t dimension) {
    double numerator = pow(M_PI, dimension / 2.0);
    double denominator = tgamma(dimension / 2.0 + 1);
    return numerator / denominator;
}

double epanechnikov_kernel(gsl_vector *pattern, double constant) {
    double dotProduct = 0.0;
    gsl_blas_ddot(pattern,  pattern, &dotProduct);

    if (dotProduct >= 1) return 0;

    return constant * (1 - dotProduct);
}

/* Normal Kernel */

void normal_prepare(size_t dimension, int numThreads) {
    g_normal_constant = epanechnikov_constant(dimension) * normal_unitVarianceConstant(dimension);
    g_normal_one_over_unit_variance_constant = 1.0 / squareRootOfTheVariance;

    g_numThreads = numThreads;

    g_scaledPattern = (gsl_vector**) malloc(numThreads * sizeof(gsl_vector*));
    for(int i = 0; i < numThreads; i++){
        g_scaledPattern[i] = gsl_vector_alloc(dimension);
    }
}

void normal_free() {
    g_normal_constant = 0.0;
    g_normal_one_over_unit_variance_constant = 0.0;

    for(int i = 0; i < g_numThreads; i++){
        gsl_vector_free(g_scaledPattern[i]);
    }
    free(g_scaledPattern);
}

double normal_pdf(gsl_vector *pattern, int pid) {
    gsl_vector* scaled_pattern = g_scaledPattern[pid];
    gsl_vector_memcpy(scaled_pattern, pattern);
    gsl_vector_scale(scaled_pattern, g_normal_one_over_unit_variance_constant);
    return epanechnikov_kernel(scaled_pattern, g_normal_constant);
}

double normal_unitVarianceConstant(size_t dimension) {
    return pow(1.0 / squareRootOfTheVariance, dimension);
}

/* Shape Adaptive Kernel */

double sa_pdf(gsl_vector* pattern, double localBandwidth){
    size_t dimension = pattern->size;

    gsl_vector_set_zero(g_sa_scaledPattern);

    // Multiply the transpose of the global inverse with the pattern
    // Since the bandwidth matrix is always symmetric we don't need to compute the transpose.
    gsl_blas_dsymv(CblasLower, 1.0, g_sa_globalInverse, pattern, 1.0, g_sa_scaledPattern);

    //Apply the local inverse
    gsl_vector_scale(g_sa_scaledPattern, 1.0 / localBandwidth);

    // Compute local scaling factor
    double localScalingFactor = computeLocalScalingFactor(g_sa_globalScalingFactor, localBandwidth, dimension);

    //Determine the result of the kernel
    return localScalingFactor * epanechnikov_kernel(g_sa_scaledPattern, g_sa_epanechnikovConstant);
}

void sa_allocate(size_t dimension){
    g_sa_globalInverse = gsl_matrix_alloc(dimension, dimension);
    g_sa_LUDecompositionH = gsl_matrix_alloc(dimension, dimension);
    g_sa_permutation = gsl_permutation_alloc(dimension);

    g_sa_scaledPattern = gsl_vector_alloc(dimension);

    sa_computeDimensionDependentConstants(dimension);
}

void sa_computeConstants(gsl_matrix *globalBandwidthMatrix){
    //Copy the global bandwidth matrix so that we can change it
    gsl_matrix_memcpy(g_sa_LUDecompositionH, globalBandwidthMatrix);

    //Scale the copy with sqrt(var(epanechnikov kernel)) to ensure unitvariance
    gsl_matrix_scale(g_sa_LUDecompositionH, squareRootOfTheVariance);

    //Compute LU decompostion
    int signum = 0;
    gsl_linalg_LU_decomp(g_sa_LUDecompositionH, g_sa_permutation, &signum);

    //Compute global inverse
    gsl_linalg_LU_invert(g_sa_LUDecompositionH, g_sa_permutation, g_sa_globalInverse);

    //Compute global scaling factor
    double determinant = gsl_linalg_LU_det(g_sa_LUDecompositionH, signum);
    g_sa_globalScalingFactor = 1.0 / determinant;
}

void sa_computeDimensionDependentConstants(size_t dimension){
    g_sa_epanechnikovConstant = epanechnikov_constant(dimension);
}

void sa_free(){
    gsl_matrix_free(g_sa_globalInverse);
    gsl_matrix_free(g_sa_LUDecompositionH);
    gsl_permutation_free(g_sa_permutation);

    gsl_vector_free(g_sa_scaledPattern);

    g_sa_globalScalingFactor = 0.0;
    g_sa_epanechnikovConstant = 0.0;
}
