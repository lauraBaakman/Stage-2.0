#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_matrix.h>
#include "gaussian.ih"

Kernel standardGaussianKernel = {
        .isShapeAdaptive = false,
        .kernel.symmetricKernel.density = normal_pdf,
        .kernel.symmetricKernel.free = normal_free,
        .kernel.symmetricKernel.prepare = normal_prepare,
};

Kernel shapeAdaptiveGaussianKernel = {
        .isShapeAdaptive = true,
        .kernel.shapeAdaptiveKernel.density = sa_pdf,
        .kernel.shapeAdaptiveKernel.allocate = sa_allocate,
        .kernel.shapeAdaptiveKernel.free = sa_free,
        .kernel.shapeAdaptiveKernel.computeConstants = sa_computeConstants,
};

static double g_standardGaussianConstant;

static gsl_matrix* g_sa_globalInverse;
static double g_sa_globalScalingFactor;
static gsl_matrix* g_sa_LUDecompositionH;
static gsl_vector* g_sa_scaledPattern;
static gsl_permutation* g_sa_permutation;

/* Normal Kernel */

double computeStandardGaussianConstant(size_t dimension){
    return pow(2 * M_PI, -1 * (double) dimension * 0.5);
}

void normal_prepare(size_t dimension, int numThreads) {
    g_standardGaussianConstant = computeStandardGaussianConstant(dimension);
}

double normal_pdf(gsl_vector *pattern, int pid) {
    double dotProduct = 0.0;
    gsl_blas_ddot(pattern,  pattern, &dotProduct);

    double density = g_standardGaussianConstant * exp(-0.5 * dotProduct);

    return  density;
}

void normal_free() {
    g_standardGaussianConstant = 0.0;
}

/* Shape Adaptive Kernel */

double sa_pdf(gsl_vector *pattern, double localBandwidth){

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
    double density = localScalingFactor * normal_pdf(g_sa_scaledPattern, 0);

    return density;
}

void sa_allocate(size_t dimension) {
    g_sa_globalInverse = gsl_matrix_alloc(dimension, dimension);
    g_sa_LUDecompositionH = gsl_matrix_alloc(dimension, dimension);
    g_sa_scaledPattern = gsl_vector_alloc(dimension);
    g_sa_permutation = gsl_permutation_alloc(dimension);

    //Compute the Standard Gaussian Constant
    sa_computeDimensionDependentConstants(dimension);
}

void sa_computeDimensionDependentConstants(size_t dimension) {

    g_standardGaussianConstant = computeStandardGaussianConstant(dimension);
}

void sa_computeConstants(gsl_matrix *globalBandwidthMatrix) {
    //Copy the global bandwidth matrix so that we can change it
    gsl_matrix_memcpy(g_sa_LUDecompositionH, globalBandwidthMatrix);

    //Compute LU decompostion
    int signum = 0;
    gsl_linalg_LU_decomp(g_sa_LUDecompositionH, g_sa_permutation, &signum);

    //Compute global inverse
    gsl_linalg_LU_invert(g_sa_LUDecompositionH, g_sa_permutation, g_sa_globalInverse);

    //Compute global scaling factor
    double determinant = gsl_linalg_LU_det(g_sa_LUDecompositionH, signum);
    g_sa_globalScalingFactor = 1.0 / determinant;

}

void sa_free() {
    g_standardGaussianConstant = 0.0;
    g_sa_globalScalingFactor = 0.0;
    gsl_matrix_free(g_sa_globalInverse);
    gsl_matrix_free(g_sa_LUDecompositionH);
    gsl_vector_free(g_sa_scaledPattern);
    gsl_permutation_free(g_sa_permutation);
}
