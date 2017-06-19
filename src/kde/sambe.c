#include "sambe.ih"

gsl_matrix* g_xs;
size_t g_numXs;

gsl_vector* g_localBandwidths;
double g_globalBandwidthFactor;
gsl_matrix* g_globalBandwidthMatrix;

ShapeAdaptiveKernel g_kernel;

gsl_vector* g_movedPattern;

size_t g_k;
gsl_matrix* g_nearestNeighbours;

int g_numThreads;

void sambe(gsl_matrix *xs,
           gsl_vector *localBandwidths, double globalBandwidth,
           KernelType kernelType, int k,
           gsl_vector *outDensities){

    prepareGlobals(xs, localBandwidths, globalBandwidth, kernelType, k);
    g_kernel.allocate(xs->size2, g_numThreads);

    double density;
    gsl_vector_view x;

    int pid = 0;
    for(size_t i = 0; i < g_numXs; i++){
        x = gsl_matrix_row(xs, i);

        density = singlePattern(&x.vector, pid);

        gsl_vector_set(outDensities, i, density);
    }

    g_kernel.free();
    freeGlobals();
}

double singlePattern(gsl_vector *x, int pid) {
    double localBandwidth, density = 0.0;

    gsl_vector_view xi;
    gsl_vector* movedPattern;
    determineGlobalKernelShape(x, pid);

    g_kernel.computeConstants(g_globalBandwidthMatrix, pid);
    
    for(size_t i = 0; i < g_numXs; i++){
        xi = gsl_matrix_row(g_xs, i);

        //x - xi
        movedPattern = gsl_subtract(x, &xi.vector, g_movedPattern);
        localBandwidth = gsl_vector_get(g_localBandwidths, i);

        density += g_kernel.density(movedPattern, localBandwidth, pid);
    }

    density /= g_numXs;

    return density;
}

void determineGlobalKernelShape(gsl_vector* x, int pid) {
    /* Compute K nearest neighbours */
    computeKNearestNeighbours(x, g_k, g_nearestNeighbours);

    /* Compute the covariance matrix of the neighbours */
    computeCovarianceMatrix(g_nearestNeighbours, g_globalBandwidthMatrix);

    /* Compute the scaling factor */
    double scalingFactor = computeScalingFactor(g_globalBandwidthFactor, g_globalBandwidthMatrix);

    /* Scale the shape matrix */
    gsl_matrix_scale(g_globalBandwidthMatrix, scalingFactor);
}

void allocateGlobals(size_t dataDimension, size_t num_xi_s, size_t k) {
    g_globalBandwidthMatrix = gsl_matrix_alloc(dataDimension, dataDimension);
    g_nearestNeighbours = gsl_matrix_alloc(k, dataDimension);
    g_movedPattern = gsl_vector_alloc(dataDimension);
}

void freeGlobals() {
    gsl_matrix_free(g_globalBandwidthMatrix);
    gsl_matrix_free(g_nearestNeighbours);
    gsl_vector_free(g_movedPattern);

    nn_free();
}

void prepareGlobals(gsl_matrix *xs,
                    gsl_vector *localBandwidths, double globalBandwidth,
                    KernelType kernelType, int k) {
    g_numThreads = 1;
    #pragma omp parallel 
    {
        g_numThreads = omp_get_num_threads();
    }
    printf("Num Threads: %d\n", g_numThreads);    

    g_kernel = selectShapeAdaptiveKernel(kernelType);

    g_xs = xs;

    g_localBandwidths = localBandwidths;
    g_globalBandwidthFactor = globalBandwidth;
    g_k = (size_t) k;

    g_numXs = xs->size1;

    size_t dimension = g_xs->size2;

    nn_prepare(xs);

    allocateGlobals(dimension, g_numXs, g_k);
}
