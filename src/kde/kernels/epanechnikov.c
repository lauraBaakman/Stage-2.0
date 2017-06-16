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

static gsl_vector** g_sa_scaledPatterns;
static gsl_matrix** g_sa_globalInverses;
static gsl_matrix** g_sa_LUDecompositionsH;
static gsl_permutation** g_sa_permutations;
static double* g_sa_globalScalingFactors;

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

double sa_pdf(gsl_vector* pattern, double localBandwidth, int pid){
    size_t dimension = pattern->size;
    gsl_vector* scaled_pattern = g_sa_scaledPatterns[pid];
    gsl_matrix* globalInverse = g_sa_globalInverses[pid];
    double globalScalingFactor = g_sa_globalScalingFactors[pid];

    gsl_vector_set_zero(scaled_pattern);

    // Multiply the transpose of the global inverse with the pattern
    // Since the bandwidth matrix is always symmetric we don't need to compute the transpose.
    gsl_blas_dsymv(CblasLower, 1.0, globalInverse, pattern, 1.0, scaled_pattern);

    //Apply the local inverse
    gsl_vector_scale(scaled_pattern, 1.0 / localBandwidth);

    // Compute local scaling factor
    double localScalingFactor = computeLocalScalingFactor(globalScalingFactor, localBandwidth, dimension);

    //Determine the result of the kernel
    return localScalingFactor * epanechnikov_kernel(scaled_pattern, g_sa_epanechnikovConstant);
}

void sa_allocate(size_t dimension, int numThreads){
    sa_computeDimensionDependentConstants(dimension);

    g_numThreads = numThreads;

    g_sa_scaledPatterns = (gsl_vector**) malloc(numThreads * sizeof(gsl_vector*));
    g_sa_globalInverses = (gsl_matrix**) malloc(numThreads * sizeof(gsl_matrix*));
    g_sa_LUDecompositionsH = (gsl_matrix**) malloc(numThreads * sizeof(gsl_matrix*));
    g_sa_permutations = (gsl_permutation**) malloc(numThreads * sizeof(gsl_permutation*));
    g_sa_globalScalingFactors = (double*) malloc(numThreads * sizeof(double));
    for(int i = 0; i < numThreads; i++){
        g_sa_scaledPatterns[i] = gsl_vector_alloc(dimension);
        g_sa_globalInverses[i] = gsl_matrix_alloc(dimension, dimension);
        g_sa_LUDecompositionsH[i] = gsl_matrix_alloc(dimension, dimension);
        g_sa_permutations[i] = gsl_permutation_alloc(dimension);
    }    
}

void sa_computeConstants(gsl_matrix *globalBandwidthMatrix, int pid){
    gsl_matrix* LUDecompositionH = g_sa_LUDecompositionsH[pid];
    gsl_permutation* permutation = g_sa_permutations[pid];
    gsl_matrix* globalInverse = g_sa_globalInverses[pid];

    //Copy the global bandwidth matrix so that we can change it
    gsl_matrix_memcpy(LUDecompositionH, globalBandwidthMatrix);

    //Scale the copy with sqrt(var(epanechnikov kernel)) to ensure unitvariance
    gsl_matrix_scale(LUDecompositionH, squareRootOfTheVariance);

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
    g_sa_epanechnikovConstant = epanechnikov_constant(dimension);
}

void sa_free(){
    for(int i = 0; i < g_numThreads; i++){
        gsl_vector_free(g_sa_scaledPatterns[i]);
        gsl_matrix_free(g_sa_globalInverses[i]);
        gsl_matrix_free(g_sa_LUDecompositionsH[i]);
        gsl_permutation_free(g_sa_permutations[i]);
    }
    free(g_sa_scaledPatterns);
    free(g_sa_globalInverses);
    free(g_sa_LUDecompositionsH);
    free(g_sa_permutations);
    free(g_sa_globalScalingFactors);

    g_sa_epanechnikovConstant = 0.0;
}
