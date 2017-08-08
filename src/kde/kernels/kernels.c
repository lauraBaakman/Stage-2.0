#include <gsl/gsl_vector_double.h>
#include "kernels.ih"
#include "kernels.h"

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
        case SHAPE_ADAPTIVE_EPANECHNIKOV:
            return shapeAdaptiveEpanechnikovKernel;
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
        case SHAPE_ADAPTIVE_EPANECHNIKOV:
            return shapeAdaptiveEpanechnikovKernel.kernel.shapeAdaptiveKernel;
        default:
            fprintf(stderr, "%d is an invalid shape adaptive kernel type.\n", type);
            exit(-1);
    }
}

double computeScalingFactor(
    double localBandwidth, double generalBandwidth, gsl_matrix* covarianceMatrix, 
    gsl_vector* eigenValues, gsl_matrix* eigenVectors
) {
    computeEigenValues(covarianceMatrix, eigenValues, eigenVectors);
    size_t dimension = eigenValues->size;

    double bandwidthTerm = log(localBandwidth * generalBandwidth);
    double eigenValuesTerm = 0.0;
    for(size_t i = 0; i < dimension; i++){
        eigenValuesTerm += log(gsl_vector_get(eigenValues, i));
    }
    return exp(bandwidthTerm - (1.0 / dimension) * eigenValuesTerm);
}