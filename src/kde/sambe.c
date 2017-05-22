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

void sambeFinalDensity(gsl_matrix* xs, gsl_matrix* xis,
                       gsl_vector* localBandwidths, double globalBandwidth,
                       ShapeAdaptiveKernel kernel,
                       gsl_vector* outDensities){
    double density;
    gsl_vector_view x;

    size_t num_xs = xs->size1;

    for(size_t i = 0; i < num_xs; i++){
        x = gsl_matrix_row(xs, i);

        density = sambeFinalDensitySinglePattern(&x.vector, xis, localBandwidths, globalBandwidth, kernel);

        gsl_vector_set(outDensities, i, density);
    }

}

double sambeFinalDensitySinglePattern(gsl_vector *x, gsl_matrix *xis, gsl_vector* localBandwidths,
                         double globalBandwidth,
                         ShapeAdaptiveKernel kernel) {

    size_t dimension = xis->size2;
    size_t dataPointCount = xis->size1;

    double localBandwidth, density = 0.0;

    gsl_vector_view xi;
    gsl_matrix* globalKernelShape = determineKernelShape(x);

    /* Prepare the evaluation of the kernel */


    /* Estimate Density */
    for(size_t xiIDX = 0; xiIDX < dataPointCount; xiIDX++){
        xi = gsl_matrix_row(xis, xiIDX);
        localBandwidth = gsl_vector_get(localBandwidths, xiIDX);

        density += evaluateKernel(x, &xi.vector, localBandwidth);
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

double evaluateKernel(gsl_vector *x, gsl_vector *xi, double localBandwidth) {
    return 42.0;
}
