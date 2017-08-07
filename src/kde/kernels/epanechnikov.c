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


static int g_numThreads;

static double sqrtFive = 2.236067977499790; // 1 / sqrt(1 / 5) == sqrt(5)
static double g_normal_constant;

static gsl_vector** g_sa_scaledPatterns;
static gsl_matrix** g_sa_globalInverses;
static gsl_matrix** g_sa_LUDecompositionsH;
static gsl_permutation** g_sa_permutations;
static double* g_sa_globalScalingFactors;

static double g_sa_epanechnikovConstant;

double computeEpanechnikovConstant(size_t dimension){
    double unitVarianceConstant = computeUnitVarianceConstant(dimension);
    double unitSphereConstant = computeUnitSphereConstant(dimension);
    return unitVarianceConstant * unitSphereConstant;    
}

double computeUnitVarianceConstant(size_t dimension){
    return pow(sqrtFive, - (int) dimension);
}

double computeUnitSphereConstant(size_t dimension){
    return (2.0 + dimension) / (2 * unitSphereVolume(dimension));
}

double unitSphereVolume(size_t dimension) {
    double numerator = pow(M_PI, dimension / 2.0);
    double denominator = tgamma(dimension / 2.0 + 1);
    return numerator / denominator;
}

double epanechnikov_kernel(gsl_vector *pattern, double constant) {
    double dotProduct = 0.0;
    gsl_blas_ddot(pattern,  pattern, &dotProduct);

    if (dotProduct >= sqrtFive) return 0;

    return constant * (1 - (1 / 5.0 * dotProduct));
}

/* Normal Kernel */

void normal_prepare(size_t dimension, int numThreads) {
    g_normal_constant = computeEpanechnikovConstant(dimension);

    g_numThreads = numThreads;
}

void normal_free() {
    //nothing to free
}

double normal_pdf(gsl_vector *pattern, int pid) {
    return epanechnikov_kernel(pattern, g_normal_constant);
}

/* Shape Adaptive Kernel */

double sa_pdf(gsl_vector* pattern, double localBandwidth, int pid){
    gsl_vector* scaled_pattern = g_sa_scaledPatterns[pid];
    gsl_matrix* globalInverse = g_sa_globalInverses[pid];
    double globalScalingFactor = g_sa_globalScalingFactors[pid];

    gsl_vector_set_zero(scaled_pattern);

    // Multiply the transpose of the global inverse with the pattern
    // Since the bandwidth matrix is always symmetric we don't need to compute the transpose.
    gsl_blas_dsymv(CblasLower, 1.0, globalInverse, pattern, 1.0, scaled_pattern);

    //Determine the result of the kernel
    return globalScalingFactor * epanechnikov_kernel(scaled_pattern, g_sa_epanechnikovConstant);
}

void sa_allocate(size_t dimension, int numThreads){
    sa_computeDimensionDependentConstants(dimension);

    g_numThreads = numThreads;

    g_sa_globalScalingFactors = (double*) malloc(numThreads * sizeof(double));

    g_sa_scaledPatterns = gsl_vectors_alloc(dimension, g_numThreads);
    g_sa_globalInverses = gsl_matrices_alloc(dimension, dimension, g_numThreads);
    g_sa_LUDecompositionsH = gsl_matrices_alloc(dimension, dimension, g_numThreads);
    g_sa_permutations = gsl_permutations_alloc(dimension, g_numThreads);
}

void sa_computeConstants(gsl_matrix *globalBandwidthMatrix, int pid){
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

void sa_computeDimensionDependentConstants(size_t dimension){
    g_sa_epanechnikovConstant = computeEpanechnikovConstant(dimension);
}

void sa_free(){
    free(g_sa_globalScalingFactors);

    gsl_vectors_free(g_sa_scaledPatterns, g_numThreads);
    gsl_matrices_free(g_sa_globalInverses, g_numThreads);
    gsl_matrices_free(g_sa_LUDecompositionsH, g_numThreads);
    gsl_permutations_free(g_sa_permutations, g_numThreads);

    g_sa_epanechnikovConstant = 0.0;
}
