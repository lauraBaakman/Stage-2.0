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
        .kernel.shapeAdaptiveKernel.factorFunction = computeGlobalConstants,
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
double shapeAdaptiveGaussianPDF(gsl_vector* pattern, double localBandwidth,
                                double globalScalingFactor, gsl_matrix * globalInverse,
                                gsl_vector* mean, gsl_matrix* cholCovmat,
                                gsl_vector* scaledPattern, gsl_vector* work, gsl_matrix* localInverse, double gaussianConstant){

    size_t dimension = globalInverse->size1;

    // Compute inverse of local bandwidth matrix
    computeLocalInverse(globalInverse, localBandwidth, localInverse);


    // Multiply the transpose of the inverse with the pattern
    // Since the bandwidth matrix is always symmetric we don't need to compute the transpose.
    gsl_blas_dsymv(CblasLower, 1.0, localInverse, pattern, 1.0, scaledPattern);

    // Compute local scaling factor
    double localScalingFactor = computeLocalScalingFactor(globalScalingFactor, localBandwidth, dimension);

    //Determine the result of the kernel
    double density = localScalingFactor * standardGaussianPDF(scaledPattern->data, (int) dimension, gaussianConstant);
    
    return density;
}

void computeGlobalConstants(Array* globalBandwidthMatrixArray, gsl_matrix *outGlobalInverse, double *outGlobalScalingFactor) {
    gsl_matrix* LUDecompH = arrayCopyToGSLMatrix(globalBandwidthMatrixArray);

    //Compute LU decompostion
    gsl_permutation* permutation = gsl_permutation_calloc((size_t) globalBandwidthMatrixArray->dimensionality);
    int signum = 0;
    gsl_linalg_LU_decomp(LUDecompH, permutation, &signum);

    //Compute global inverse
    gsl_linalg_LU_invert(LUDecompH, permutation, outGlobalInverse);

    //Compute global scaling factor
    *outGlobalScalingFactor = 1.0 / gsl_linalg_LU_det(LUDecompH, signum);

    //Free memory
    gsl_permutation_free(permutation);
}

double computeLocalScalingFactor(double globalScalingFactor, double localBandwidth, size_t dimension) {
    double localScalingFactor = (1.0 / pow(localBandwidth, dimension)) * globalScalingFactor;
    return localScalingFactor;
}

void computeLocalInverse(gsl_matrix* globalInverse, double localBandwidth, gsl_matrix* outLocalInverse){
    gsl_matrix_memcpy(outLocalInverse, globalInverse);
    gsl_matrix_scale(outLocalInverse, 1.0 / localBandwidth);
}

/* Utilities */

double dotProduct(double *a, double *b, int length) {
    double dotProduct = 0;
    for ( int i = 0; i < length; ++i ) {
        dotProduct += (a[i] * b[i]);
    }
    return dotProduct;
}