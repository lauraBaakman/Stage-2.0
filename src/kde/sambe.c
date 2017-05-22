#include <gsl/gsl_matrix.h>
#include "utils/distancematrix.h"
#include "utils/knn.h"
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

gsl_matrix* g_xs;
gsl_vector* g_localBandwidths;
double g_globalBandwidth;
ShapeAdaptiveKernel g_kernel;


gsl_matrix* g_globalKernelShape;

int g_k;
gsl_matrix* g_distanceMatrix;
gsl_matrix* g_nearestNeighbours;

void sambeFinalDensity(gsl_matrix* xs, gsl_matrix* xis,
                       gsl_vector* localBandwidths, double globalBandwidth,
                       ShapeAdaptiveKernel kernel,
                       gsl_vector* outDensities){

    /* Take care of the globals */
    g_xs = xs;

    g_localBandwidths = localBandwidths;
    g_globalBandwidth = globalBandwidth;
    g_kernel = kernel;
    g_k = 1;

    size_t dimension = g_xs->size2;
    size_t num_x_s = xs->size1;

    allocateGlobals(dimension, num_x_s, g_k);

    computeDistanceMatrix(g_xs, g_distanceMatrix);

    /* Do the computations */
    double density;
    gsl_vector_view x;

    size_t num_xs = xs->size1;

    for(size_t i = 0; i < num_xs; i++){
        x = gsl_matrix_row(xs, i);

        density = sambeFinalDensitySinglePattern(&x.vector, i);

        gsl_vector_set(outDensities, i, density);
    }

    /* Free memory */
    freeGlobals();

}

double sambeFinalDensitySinglePattern(gsl_vector *x, size_t xIdx) {

    size_t dimension = g_xs->size2;
    size_t dataPointCount = g_xs->size1;

    double localBandwidth, density = 0.0;

    gsl_vector_view xi;
    determineGlobalKernelShape(xIdx);

    /* Prepare the evaluation of the kernel */


    /* Estimate Density */
    for(size_t xiIDX = 0; xiIDX < dataPointCount; xiIDX++){
        xi = gsl_matrix_row(g_xs, xiIDX);
        localBandwidth = gsl_vector_get(g_localBandwidths, xiIDX);

        density += evaluateKernel(x, &xi.vector, localBandwidth);
    }

    density /= dataPointCount;

    /* free memory */

    return density;
}

void determineGlobalKernelShape(size_t patternIdx) {
    /* Compute K nearest neighbours */
    compute_k_nearest_neighbours(g_k, (int)patternIdx,
                                 g_xs, g_distanceMatrix,
                                 g_nearestNeighbours);

    /* Compute the covariance matrix of the neighbours */

    /* Compute the scaling factor */

    /* Scale the shape matrix */

}

double evaluateKernel(gsl_vector *x, gsl_vector *xi, double localBandwidth) {
    return 42.0;
}

void allocateGlobals(size_t dataDimension, size_t num_xi_s, int k) {
    g_globalKernelShape = gsl_matrix_alloc(dataDimension, dataDimension);
    g_distanceMatrix = gsl_matrix_alloc(num_xi_s, num_xi_s);
    g_nearestNeighbours = gsl_matrix_alloc(k, dataDimension);
}

void freeGlobals() {
    gsl_matrix_free(g_globalKernelShape);
    gsl_matrix_free(g_distanceMatrix);
    gsl_matrix_free(g_nearestNeighbours);
}
