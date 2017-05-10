#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_vector.h>
#include "kernels.ih"
#include "../utils/eigenvalues.h"
#include "../../../../../../../usr/local/include/gsl/gsl_vector_double.h"
#include "kernels.h"

Kernel standardGaussianKernel = {
        .isSymmetric = true,
        .isShapeAdaptive = false,
        .kernel.symmetricKernel.densityFunction = standardGaussianPDF,
        .kernel.symmetricKernel.factorFunction = standardGaussianConstant,
};

Kernel epanechnikovKernel = {
        .isSymmetric = true,
        .isShapeAdaptive = false,
        .kernel.symmetricKernel.densityFunction = epanechnikovPDF,
        .kernel.symmetricKernel.factorFunction = epanechnikovConstant,
};

Kernel testKernel = {
        .isSymmetric = true,
        .isShapeAdaptive = false,
        .kernel.symmetricKernel.densityFunction = testKernelPDF,
        .kernel.symmetricKernel.factorFunction = testKernelConstant,
};

Kernel gaussianKernel = {
        .isSymmetric = false,
        .isShapeAdaptive = false,
        .kernel.aSymmetricKernel.densityFunction = gaussianPDF,
        .kernel.aSymmetricKernel.factorFunction= gaussianConstant,
};

Kernel shapeAdaptiveGaussianKernel = {
        .isSymmetric = false,
        .isShapeAdaptive = true,
        .kernel.shapeAdaptiveKernel.densityFunction = shapeAdaptiveGaussianPDF,
        .kernel.shapeAdaptiveKernel.factorFunction= shapeAdaptiveConstant,
};


Kernel selectKernel(KernelType type) {
    switch (type) {
        case EPANECHNIKOV:
            return epanechnikovKernel;
        case STANDARD_GAUSSIAN:
            return standardGaussianKernel;
        case TEST:
            return testKernel;
        case GAUSSIAN:
            return gaussianKernel;
        case SHAPE_ADAPTIVE_GAUSSIAN:
            return shapeAdaptiveGaussianKernel;
        default:
            fprintf(stderr, "%d is an invalid kernel type.\n", type);
            exit(-1);
    }
}

SymmetricKernel selectSymmetricKernel(KernelType type) {
    switch (type) {
        case EPANECHNIKOV:
            return epanechnikovKernel.kernel.symmetricKernel;
        case STANDARD_GAUSSIAN:
            return standardGaussianKernel.kernel.symmetricKernel;
        case TEST:
            return testKernel.kernel.symmetricKernel;
        default:
            fprintf(stderr, "%d is an invalid  symmetric kernel type.\n", type);
            exit(-1);
    }
}

ASymmetricKernel selectASymmetricKernel(KernelType type) {
    switch (type) {
        case GAUSSIAN:
            return gaussianKernel.kernel.aSymmetricKernel;
        default:
            fprintf(stderr, "%d is an invalid asymmetric kernel type.\n", type);
            exit(-1);
    }
}

ShapeAdaptiveKernel selectShapeAdaptiveKernel(KernelType type){
    switch(type) {
        case SHAPE_ADAPTIVE_GAUSSIAN:
            return shapeAdaptiveGaussianKernel.kernel.shapeAdaptiveKernel;
        default:
            fprintf(stderr, "%d is an invalid shape adaptive kernel type.\n", type);
            exit(-1);
    }
}

double computeScalingFactor(double generalBandwidth, gsl_matrix_view covarianceMatrix) {
    gsl_vector* eigenvalues = computeEigenValues2(&covarianceMatrix.matrix);
    size_t dimension = eigenvalues->size;

    double generalBandWidthTerm = log(generalBandwidth);
    double eigenValuesTerm = 0.0;
    for(size_t i = 0; i < dimension; i++){
        eigenValuesTerm += log(gsl_vector_get(eigenvalues, i));
    }
    gsl_vector_free(eigenvalues);
    return exp(generalBandWidthTerm - (1.0 / dimension) * eigenValuesTerm);
}


/* Symmetric Kernels */

double standardGaussianConstant(int patternDimensionality) {
    return pow(2 * M_PI, -1 * patternDimensionality * 0.5);
}

double standardGaussianPDF(double *pattern, int patternDimensionality, double constant) {
    double dotProduct = 0.0;
    for ( int i = 0; i < patternDimensionality; i++ ) {
        dotProduct += pattern[i] * pattern[i];
    }
    return constant * exp(-0.5 * dotProduct);
}

double epanechnikovConstant(int dimensionality) {
    double numerator = pow(M_PI, dimensionality / 2.0);
    double denominator = gamma(dimensionality / 2.0 + 1);
    return 2 * (numerator / denominator);
}

double epanechnikovPDF(double *data, int dimensionality, double constant) {
    double patternDotPattern = dotProduct(data, data, dimensionality);
    if (patternDotPattern >= 1) {
        return 0;
    }
    double numerator = (double) dimensionality + 2;
    return (numerator / constant) * (1 - patternDotPattern);
}

double testKernelConstant(int patternDimensionality) {
    return 1.0 / patternDimensionality;
}

double testKernelPDF(double *data, int dimensionality, double constant) {
    double density = 0;
    for ( int i = 0; i < dimensionality; i++ ) {
        density += data[i];
    }
    double mean = density * constant;
    return fabs(mean);
}


/* Asymmetric Kernels */

gsl_matrix * gaussianConstant(Array* covarianceMatrix) {
    gsl_matrix* choleskyDecomposition = arrayCopyToGSLMatrix(covarianceMatrix);
    gsl_linalg_cholesky_decomp1(choleskyDecomposition);
    return choleskyDecomposition;
}

double gaussianPDF(gsl_vector * pattern, gsl_vector * mean, gsl_matrix *choleskyFactorCovarianceMatrix) {
    double density;
    gsl_vector* work = gsl_vector_alloc(mean->size);

    gsl_ran_multivariate_gaussian_pdf(pattern, mean, choleskyFactorCovarianceMatrix, &density, work);

    gsl_vector_free(work);
    return density;
}

/* Shape Adaptive Kernels */

gsl_matrix* shapeAdaptiveConstant(Array* covarianceMatrix){
    gsl_matrix* globalBandwidthMatrixCholeskyFactorization = arrayCopyToGSLMatrix(covarianceMatrix);

    gsl_linalg_cholesky_decomp1(globalBandwidthMatrixCholeskyFactorization);

    return globalBandwidthMatrixCholeskyFactorization;
}

double shapeAdaptiveGaussianPDF(gsl_vector* pattern, double localBandwidth, gsl_matrix * bandwidthMatrix){
    //Compute local bandwidth matrix
    gsl_matrix_scale(bandwidthMatrix, localBandwidth);

    //Compute the LU factorization of the local bandwidth matrix
    gsl_matrix* luDecomposition = gsl_matrix_alloc(bandwidthMatrix->size1, bandwidthMatrix->size2);
    gsl_matrix_memcpy(luDecomposition, bandwidthMatrix);

    gsl_permutation* permutation = gsl_permutation_calloc(luDecomposition->size2);
    int signum = 0;
    gsl_linalg_LU_decomp(luDecomposition, permutation, &signum);

    // Compute inverse of local bandwidth matrix
    gsl_matrix* inverse = gsl_matrix_alloc(bandwidthMatrix->size1, bandwidthMatrix->size2);
    gsl_linalg_LU_invert(luDecomposition,permutation, inverse);

    // Compute determinant of local bandwidth matrix
    double determinant = gsl_linalg_LU_det(luDecomposition, signum);

    // Multiply the inverse with the pattern INPLACE!!
    gsl_matrix_transpose(inverse);
    gsl_vector* scaled_pattern = gsl_vector_calloc(pattern->size);
    gsl_blas_dsymv(CblasLower, 1.0, inverse, pattern, 1.0, scaled_pattern);

    //Evaluate the pdf
    gsl_vector* mean = gsl_vector_calloc(bandwidthMatrix->size1);
    gsl_matrix* covarianceMatrix = gsl_matrix_alloc(bandwidthMatrix->size1, bandwidthMatrix->size2);
    gsl_matrix_set_identity(covarianceMatrix);

    gsl_matrix* choleskyDecompositionCovarianceMatrix = gsl_matrix_alloc(covarianceMatrix->size1, covarianceMatrix->size2);
    gsl_matrix_memcpy(choleskyDecompositionCovarianceMatrix, covarianceMatrix);

    gsl_linalg_cholesky_decomp1(choleskyDecompositionCovarianceMatrix);

    gsl_vector* work = gsl_vector_alloc(mean->size);

    double density = 1.0; //Skipping initialization of this value breaks the evaluation of the pdf
    gsl_ran_multivariate_gaussian_pdf(scaled_pattern, mean, choleskyDecompositionCovarianceMatrix, &density, work);

    //Determine the result of the kernel.
    double scalingFactor = 1.0 / determinant;
    density *= scalingFactor;

    //Free memory
    gsl_matrix_free(luDecomposition);
    gsl_matrix_free(inverse);
    gsl_vector_free(mean);
    gsl_vector_free(scaled_pattern);
    gsl_matrix_free(covarianceMatrix);
    gsl_matrix_free(choleskyDecompositionCovarianceMatrix);
    gsl_vector_free(work);
    gsl_permutation_free(permutation);

    return density;
}


/* Utilities */

double dotProduct(double *a, double *b, int length) {
    double dotProduct = 0;
    for ( int i = 0; i < length; ++i ) {
        dotProduct += (a[i] * b[i]);
    }
    return dotProduct;
}