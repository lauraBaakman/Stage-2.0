#include "sambe.ih"

double sambeFinalDensity(double *pattern, Array *datapoints,
                         double globalBandwidth,
                         ShapeAdaptiveKernel kernel) {

    size_t dimension = (size_t) datapoints->dimensionality;

    double* currentDataPoint = datapoints->data;

    gsl_matrix* globalKernelShape = determineKernelShape(pattern);

    /* Prepare the evaluation of the kernel */
    double pdfConstant;
    double globalScalingFactor;
    gsl_matrix* globalInverse = gsl_matrix_alloc(dimension, dimension);
    kernel.factorFunction(globalKernelShape, globalInverse, &globalScalingFactor, &pdfConstant);



    /* Free memory */
    gsl_matrix_free(globalInverse);
}

gsl_matrix *determineKernelShape(double *pattern) {
    return NULL;
}

double finalDensityEstimatePattern(double *pattern) {
    return 37.0;
}
