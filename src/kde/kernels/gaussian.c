#include <gsl/gsl_vector_double.h>
#include "gaussian.ih"

Kernel standardGaussianKernel = {
        .isShapeAdaptive = false,
        .kernel.symmetricKernel.densityFunction = normal_pdf,
        .kernel.symmetricKernel.factorFunction = normal_constant_compute,
        .kernel.symmetricKernel.free = normal_free,
        .kernel.symmetricKernel.prepare = normal_prepare,
};

Kernel shapeAdaptiveGaussianKernel = {
        .isShapeAdaptive = true,
        .kernel.shapeAdaptiveKernel.densityFunction = shapeAdaptiveGaussianPDF,
        .kernel.shapeAdaptiveKernel.factorFunction = shapeAdaptiveGaussianConstants,
};

/* Normal Kernel */

static double g_normal_constant;

double normal_constant_compute(int patternDimensionality){
    return pow(2 * M_PI, -1 * patternDimensionality * 0.5);
}

double standardGaussianPDF(gsl_vector* pattern, double constant){
    double dotProduct = 0.0;
    gsl_blas_ddot(pattern,  pattern, &dotProduct);
    return constant * exp(-0.5 * dotProduct);
}

void normal_prepare(size_t dimension) {
    g_normal_constant = normal_constant_compute(dimension);
}

double normal_pdf(gsl_vector *pattern) {
    normal_prepare(pattern->size);

    double dotProduct = 0.0;
    gsl_blas_ddot(pattern,  pattern, &dotProduct);

    double density = g_normal_constant * exp(-0.5 * dotProduct);

    return  density;
}

void normal_free() {
    g_normal_constant = 0.0;
}

/* Shape Adaptive Kernel */

double shapeAdaptiveGaussianPDF(gsl_vector* pattern, double localBandwidth,
                                double globalScalingFactor, gsl_matrix * globalInverse, double gaussianConstant,
                                gsl_vector* scaledPattern){

    size_t dimension = globalInverse->size1;

    // Multiply the transpose of the global inverse with the pattern
    // Since the bandwidth matrix is always symmetric we don't need to compute the transpose.
    gsl_blas_dsymv(CblasLower, 1.0, globalInverse, pattern, 1.0, scaledPattern);

    //Apply the local inverse
    gsl_vector_scale(scaledPattern, 1.0 / localBandwidth);

    // Compute local scaling factor
    double localScalingFactor = computeLocalScalingFactor(globalScalingFactor, localBandwidth, dimension);

    //Determine the result of the kernel
    double density = localScalingFactor * standardGaussianPDF(scaledPattern, gaussianConstant);

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
    *outPDFConstant = normal_constant_compute(dimension);

    //Free memory
    gsl_permutation_free(permutation);
    gsl_matrix_free(LUDecompH);
}
