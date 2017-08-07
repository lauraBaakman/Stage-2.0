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

static gsl_matrix** g_sa_globalInverses;
static double* g_sa_globalScalingFactors;
static gsl_matrix** g_sa_LUDecompositionsH;
static gsl_vector** g_sa_scaledPatterns;
static gsl_permutation** g_sa_permutations;

static int g_numThreads;

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

double sa_pdf(gsl_vector *pattern, double localBandwidth, int pid){
    gsl_vector* scaledPattern = g_sa_scaledPatterns[pid];
    gsl_matrix* globalInverse = g_sa_globalInverses[pid];
    double globalScalingFactor = g_sa_globalScalingFactors[pid];

    gsl_vector_set_zero(scaledPattern);

    // Multiply the transpose of the global inverse with the pattern
    // Since the bandwidth matrix is always symmetric we don't need to compute the transpose.
    gsl_blas_dsymv(CblasLower, 1.0, globalInverse, pattern, 1.0, scaledPattern);

    //Determine the result of the kernel
    double density = globalScalingFactor * normal_pdf(scaledPattern, 0);

    return density;
}

void sa_allocate(size_t dimension, int numThreads) {
    g_numThreads = numThreads;

    g_sa_globalInverses = gsl_matrices_alloc(dimension, dimension, numThreads);
    g_sa_LUDecompositionsH = gsl_matrices_alloc(dimension, dimension, numThreads);
    g_sa_scaledPatterns = gsl_vectors_alloc(dimension, numThreads);
    g_sa_permutations = gsl_permutations_alloc(dimension, numThreads);

    g_sa_globalScalingFactors = (double*) malloc(numThreads * sizeof(double));

    //Compute the Standard Gaussian Constant
    sa_computeDimensionDependentConstants(dimension);
}

void sa_computeDimensionDependentConstants(size_t dimension) {

    g_standardGaussianConstant = computeStandardGaussianConstant(dimension);
}

void sa_computeConstants(gsl_matrix *globalBandwidthMatrix, int pid) {
    gsl_matrix* LUDecompositionH = g_sa_LUDecompositionsH[pid];
    gsl_permutation* permutation = g_sa_permutations[pid];
    gsl_matrix* globalInverse = g_sa_globalInverses[pid];    

    //Copy the global bandwidth matrix so that we can change it
    gsl_matrix_memcpy(LUDecompositionH, globalBandwidthMatrix);

    //Compute LU decompostion
    int signum = 0;
    gsl_linalg_LU_decomp(LUDecompositionH, permutation, &signum);

    //Compute global inverse
    gsl_linalg_LU_invert(LUDecompositionH, permutation, globalInverse);

    //Compute global scaling factor
    double determinant = gsl_linalg_LU_det(LUDecompositionH, signum);
    g_sa_globalScalingFactors[pid] = 1.0 / determinant;
}

void sa_free() {
    g_standardGaussianConstant = 0.0;
    gsl_matrices_free(g_sa_globalInverses, g_numThreads);
    gsl_matrices_free(g_sa_LUDecompositionsH, g_numThreads);
    gsl_vectors_free(g_sa_scaledPatterns, g_numThreads);
    gsl_permutations_free(g_sa_permutations, g_numThreads);
    free(g_sa_globalScalingFactors);
}
