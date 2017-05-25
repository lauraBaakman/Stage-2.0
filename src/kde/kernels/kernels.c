#include <gsl/gsl_vector_double.h>
#include "kernels.ih"

Kernel testKernel = {
        .isShapeAdaptive = false,
        .kernel.symmetricKernel.densityFunction = testKernelPDF,
        .kernel.symmetricKernel.factorFunction = testKernelConstant,
};


Kernel selectKernel(KernelType type) {
    switch (type) {
        case EPANECHNIKOV:
            return epanechnikovKernel;
        case STANDARD_GAUSSIAN:
            return standardGaussianKernel;
        case TEST:
            return testKernel;
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

ShapeAdaptiveKernel selectShapeAdaptiveKernel(KernelType type){
    switch(type) {
        case SHAPE_ADAPTIVE_GAUSSIAN:
            return shapeAdaptiveGaussianKernel.kernel.shapeAdaptiveKernel;
        default:
            fprintf(stderr, "%d is an invalid shape adaptive kernel type.\n", type);
            exit(-1);
    }
}

double computeScalingFactor(double generalBandwidth, gsl_matrix* covarianceMatrix) {
    gsl_vector* eigenvalues = gsl_vector_alloc(covarianceMatrix->size1);
    computeEigenValues(covarianceMatrix, eigenvalues);
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
double testKernelConstant(int patternDimensionality) {
    return 1.0 / patternDimensionality;
}

double testKernelPDF(gsl_vector* pattern, double constant) {
    double density = 0;
    for ( size_t i = 0; i < pattern->size; i++ ) {
        density += pattern->data[i];
    }
    double mean = density * constant;
    return fabs(mean);
}

double computeLocalScalingFactor(double globalScalingFactor, double localBandwidth, size_t dimension) {
    double localScalingFactor = (1.0 / pow(localBandwidth, dimension)) * globalScalingFactor;
    return localScalingFactor;
}