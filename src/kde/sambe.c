#include "sambe.ih"

//double sambeFinalDensity(double *pattern, Array *datapoints,
//                         double globalBandwidth,
//                         ShapeAdaptiveKernel kernel) {
//
//    size_t dimension = (size_t) datapoints->dimensionality;
//
//    double* currentDataPoint = datapoints->data;
//
//    gsl_matrix* globalKernelShape = determineKernelShape(pattern);
//
//    /* Prepare the evaluation of the kernel */
//    double pdfConstant;
//    double globalScalingFactor;
//    gsl_matrix* globalInverse = gsl_matrix_alloc(dimension, dimension);
//    kernel.factorFunction(globalKernelShape, globalInverse, &globalScalingFactor, &pdfConstant);
//
//    /* Evalute
//    double density = 0;
//    for(int i = 0;
//        i < datapoints->length;
//        ++i, currentDataPoint+= datapoints->rowStride)
//    {
//        density += finalDensityEstimatePattern(currentDataPoint);
//    }
//
//    density /= datapoints->length;
//
//    /* Free memory */
//    gsl_matrix_free(globalInverse);
//
//    return density;
//}

double sambeFinalDensity(gsl_vector *pattern, gsl_matrix *datapoints, gsl_vector* localBandwidths,
                         double globalBandwidth,
                         ShapeAdaptiveKernel kernel) {
    gsl_vector_print(stdout, pattern);
    return 0;
}

gsl_matrix *determineKernelShape(gsl_vector *pattern) {
    return NULL;
}

double finalDensityEstimatePattern(gsl_vector *pattern) {
    return 37;
}
