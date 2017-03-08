#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include "kernels.ih"

Kernel standardGaussianKernel = {
        .isSymmetric = true,
        .kernel.symmetricKernel.densityFunction = standardGaussianPDF,
        .kernel.symmetricKernel.factorFunction = standardGaussianConstant,
        .kernel.symmetricKernel.scalingFactorFunction = standardGaussianScalingFactor,
};

Kernel epanechnikovKernel = {
        .isSymmetric = true,
        .kernel.symmetricKernel.densityFunction = epanechnikovPDF,
        .kernel.symmetricKernel.factorFunction = epanechnikovConstant,
        .kernel.symmetricKernel.scalingFactorFunction = epanechnikovScalingFactor,
};

Kernel testKernel = {
        .isSymmetric = true,
        .kernel.symmetricKernel.densityFunction = testKernelPDF,
        .kernel.symmetricKernel.factorFunction = testKernelConstant,
        .kernel.symmetricKernel.scalingFactorFunction = testKernelScalingFactor,
};

Kernel gaussianKernel = {
        .isSymmetric = false,
        .kernel.aSymmetricKernel.densityFunction = gaussianPDF,
        .kernel.aSymmetricKernel.factorFunction= gaussianConstant,
        .kernel.aSymmetricKernel.scalingFactorFunction = gaussianScalingFactor,
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

double standardGaussianScalingFactor(double generalBandwidth) {
    return  generalBandwidth * sqrt(generalBandwidth);
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

double epanechnikovScalingFactor(double generalBandwidth) {
    fprintf(stderr, "The function epanechnikovPDFScalingFactor is not implemented.");
    exit(-1);
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

double testKernelScalingFactor(double generalBandwidth) {
    return 0.5;
}

/* Asymmetric Kernels */

gsl_matrix *gaussianConstant(Array* covarianceMatrix) {
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

double gaussianScalingFactor(double generalBandwidth, gsl_vector *eigenValues) {
    return 42.0;
}

double dotProduct(double *a, double *b, int length) {
    double dotProduct = 0;
    for ( int i = 0; i < length; ++i ) {
        dotProduct += (a[i] * b[i]);
    }
    return dotProduct;
}