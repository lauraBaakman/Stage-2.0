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
        .kernel.shapeAdaptiveKernel.densityFunction = shapeAdaptiveGaussianPDF,
        .kernel.shapeAdaptiveKernel.factorFunction = shapeAdaptiveGaussianConstants,
};

static double g_standardGaussianConstant;

static gsl_matrix* g_sa_globalInverse;
static double g_sa_globalScalingFactor;

/* Normal Kernel */

double computeStandardGaussianConstant(size_t dimension){
    return pow(2 * M_PI, -1 * (double) dimension * 0.5);
}

void normal_prepare(size_t dimension) {
    g_standardGaussianConstant = computeStandardGaussianConstant(dimension);
}

double normal_pdf(gsl_vector *pattern) {
    double dotProduct = 0.0;
    gsl_blas_ddot(pattern,  pattern, &dotProduct);

    double density = g_standardGaussianConstant * exp(-0.5 * dotProduct);

    return  density;
}

void normal_free() {
    g_standardGaussianConstant = 0.0;
}

/* Shape Adaptive Kernel */

double shapeAdaptiveGaussianPDF(gsl_vector* pattern, double localBandwidth,
                                double globalScalingFactor, gsl_matrix * globalInverse, double gaussianConstant,
                                gsl_vector* scaledPattern, gsl_matrix* globalBandwidthMatrix){

    size_t dimension = pattern->size;

    sa_allocate(dimension);
    sa_compute_constants(globalBandwidthMatrix);

    // Multiply the transpose of the global inverse with the pattern
    // Since the bandwidth matrix is always symmetric we don't need to compute the transpose.
    gsl_blas_dsymv(CblasLower, 1.0, g_sa_globalInverse, pattern, 1.0, scaledPattern);

    //Apply the local inverse
    gsl_vector_scale(scaledPattern, 1.0 / localBandwidth);

    // Compute local scaling factor
    double localScalingFactor = computeLocalScalingFactor(g_sa_globalScalingFactor, localBandwidth, dimension);

    //Determine the result of the kernel
    double density = localScalingFactor * normal_pdf(scaledPattern);

    sa_free();

    return density;
}

void shapeAdaptiveGaussianConstants(gsl_matrix *globalBandwidthMatrix, gsl_matrix *outGlobalInverse,
                                    double *outGlobalScalingFactor, double *outPDFConstant) {

    size_t dimension = globalBandwidthMatrix->size1;

    gsl_matrix* LUDecompH = gsl_matrix_alloc(dimension, dimension);
    gsl_matrix_memcpy(LUDecompH, globalBandwidthMatrix);

    //Compute LU decompostion
    gsl_permutation* permutation = gsl_permutation_calloc(dimension);
    int signum = 0;
    gsl_linalg_LU_decomp(LUDecompH, permutation, &signum);

    //Compute global inverse
    gsl_linalg_LU_invert(LUDecompH, permutation, outGlobalInverse);

    //Compute global scaling factor
    double determinant = gsl_linalg_LU_det(LUDecompH, signum);
    *outGlobalScalingFactor = 1.0 / determinant;

    //Compute the pdfConstant
    *outPDFConstant = computeStandardGaussianConstant(dimension);

    //Free memory
    gsl_permutation_free(permutation);
    gsl_matrix_free(LUDecompH);
}

void sa_allocate(size_t dimension) {
    g_sa_globalInverse = gsl_matrix_alloc(dimension, dimension);
}

void sa_compute_constants(gsl_matrix *globalBandwidthMatrix) {
    size_t dimension = globalBandwidthMatrix->size1;

    //Compute the Standard Gaussian Constant
    g_standardGaussianConstant = computeStandardGaussianConstant(dimension);

    //Allocate memory for the LU decomposition
    gsl_matrix* LUDecompH = gsl_matrix_alloc(dimension, dimension);
    gsl_matrix_memcpy(LUDecompH, globalBandwidthMatrix);

    //Compute LU decompostion
    gsl_permutation* permutation = gsl_permutation_calloc(dimension);
    int signum = 0;
    gsl_linalg_LU_decomp(LUDecompH, permutation, &signum);

    //Compute global inverse
    gsl_linalg_LU_invert(LUDecompH, permutation, g_sa_globalInverse);

    //Compute global scaling factor
    double determinant = gsl_linalg_LU_det(LUDecompH, signum);
    g_sa_globalScalingFactor = 1.0 / determinant;

    //Free Memory
    gsl_matrix_free(LUDecompH);
}

void sa_free() {
    g_standardGaussianConstant = 0.0;
    g_sa_globalScalingFactor = 0.0;
    gsl_matrix_free(g_sa_globalInverse);
}
