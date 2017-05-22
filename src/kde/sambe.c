#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector_double.h>
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
//    /* Evaluate
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

    size_t dimension = datapoints->size2;
    size_t dataPointCount = datapoints->size1;

    double localBandwidth, density = 0.0;

    gsl_vector_view xi;
    gsl_matrix* globalKernelShape = determineKernelShape(pattern);

    /* Prepare the evaluation of the kernel */


    /* Estimate Density */
    for(size_t xiIDX = 0; xiIDX < dataPointCount; xiIDX++){
        xi = gsl_matrix_row(datapoints, xiIDX);
        localBandwidth = gsl_vector_get(localBandwidths, xiIDX);

        density += finalDensityEstimatePattern(pattern, &xi.vector, localBandwidth);
    }

    density /= dataPointCount;

    /* free memory */
    gsl_matrix_free(globalKernelShape);

    return density;
}

gsl_matrix *determineKernelShape(gsl_vector *pattern) {
    size_t dimension = pattern->size;
    gsl_matrix* shape = gsl_matrix_alloc(dimension, dimension);

    return shape;
}

double finalDensityEstimatePattern(gsl_vector *pattern, gsl_vector* xi, double localBandwidth) {
    return 37;
}
