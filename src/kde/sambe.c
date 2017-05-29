#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector_double.h>
#include "sambe.ih"

gsl_matrix* g_xs;

gsl_vector* g_localBandwidths;

double g_globalBandwidth;
gsl_matrix* g_globalBandwidthMatrix;

ShapeAdaptiveKernel g_kernel;

size_t g_numXs;

gsl_vector* g_movedPattern;

int g_k;
gsl_matrix* g_distanceMatrix;
gsl_matrix* g_nearestNeighbours;

void sambeFinalDensity(gsl_matrix* xs,
                       gsl_vector* localBandwidths, double globalBandwidth,
                       ShapeAdaptiveKernel kernel, int k,
                       gsl_vector* outDensities){

    prepareGlobals(xs, localBandwidths, globalBandwidth, kernel, k);

    kernel.allocate(xs->size2);

    double density;
    gsl_vector_view x;

    for(size_t i = 0; i < g_numXs; i++){
        x = gsl_matrix_row(xs, i);

        density = finalDensitySinglePattern(&x.vector, i);

        gsl_vector_set(outDensities, i, density);
    }

    kernel.free();
    freeGlobals();
}

double finalDensitySinglePattern(gsl_vector *x, size_t xIdx) {
    double localBandwidth, density = 0.0;

    gsl_vector_view xi;
    gsl_vector* movedPattern;
    determineGlobalKernelShape(xIdx);
    g_kernel.computeConstants(g_globalBandwidthMatrix);

    for(size_t i = 0; i < g_numXs; i++){
        xi = gsl_matrix_row(g_xs, i);

        //x - xi
        movedPattern = substract(x, &xi.vector, g_movedPattern);
        localBandwidth = gsl_vector_get(g_localBandwidths, i);

        density += g_kernel.density(movedPattern, localBandwidth);
    }

    density /= g_numXs;

    return density;
}

void determineGlobalKernelShape(size_t patternIdx) {
    /* Compute K nearest neighbours */
    computeKNearestNeighbours(g_k, (int) patternIdx,
                              g_xs, g_distanceMatrix,
                              g_nearestNeighbours);

    /* Compute the covariance matrix of the neighbours */
    computeCovarianceMatrix(g_nearestNeighbours, g_globalBandwidthMatrix);

    /* Compute the scaling factor */
    double scalingFactor = computeScalingFactor(g_globalBandwidth, g_globalBandwidthMatrix);

    /* Scale the shape matrix */
    gsl_matrix_scale(g_globalBandwidthMatrix, scalingFactor);
}

gsl_vector *substract(gsl_vector *termA, gsl_vector *termB, gsl_vector *result) {
    gsl_vector_memcpy(result, termA);
    gsl_vector_sub(result, termB);

    return result;
}

void allocateGlobals(size_t dataDimension, size_t num_xi_s, int k) {
    g_globalBandwidthMatrix = gsl_matrix_alloc(dataDimension, dataDimension);
    g_distanceMatrix = gsl_matrix_alloc(num_xi_s, num_xi_s);
    g_nearestNeighbours = gsl_matrix_alloc(k, dataDimension);
    g_movedPattern = gsl_vector_alloc(dataDimension);
}

void freeGlobals() {
    gsl_matrix_free(g_globalBandwidthMatrix);
    gsl_matrix_free(g_distanceMatrix);
    gsl_matrix_free(g_nearestNeighbours);
    gsl_vector_free(g_movedPattern);
}

void prepareGlobals(gsl_matrix *xs,
                    gsl_vector *localBandwidths, double globalBandwidth,
                    ShapeAdaptiveKernel kernel, int k) {
    g_xs = xs;

    g_localBandwidths = localBandwidths;
    g_globalBandwidth = globalBandwidth;
    g_kernel = kernel;
    g_k = k;

    g_numXs = xs->size1;

    size_t dimension = g_xs->size2;

    allocateGlobals(dimension, g_numXs, g_k);

    computeDistanceMatrix(g_xs, g_distanceMatrix);
}
