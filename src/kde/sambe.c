#include "sambe.ih"

gsl_matrix* g_xs;
size_t g_numXs;

gsl_vector* g_localBandwidths;
double g_globalBandwidthFactor;
gsl_matrix** g_globalBandwidthMatrices;

ShapeAdaptiveKernel g_kernel;

gsl_vector** g_movedPatterns;

size_t g_k;
gsl_matrix** g_nearestNeighbourMatrices;

static int g_numThreads;

void sambe(gsl_matrix *xs,
           gsl_vector *localBandwidths, double globalBandwidth,
           KernelType kernelType, int k,
           gsl_vector *densities){

    prepareGlobals(xs, localBandwidths, globalBandwidth, kernelType, k);    

    #pragma omp parallel shared(g_numXs, xs, densities)
    {
        int pid = omp_get_thread_num();
        double density;
        gsl_vector_view x;

        #pragma omp for
        for(size_t i = 0; i < g_numXs; i++){
            x = gsl_matrix_row(xs, i);

            density = singlePattern(&x.vector, pid);

            gsl_vector_set(densities, i, density);
        }        
    }

    freeGlobals();
}

double singlePattern(gsl_vector *x, int pid) {
    gsl_matrix* globalBandwidthMatrix = g_globalBandwidthMatrices[pid];
    gsl_vector* movedPattern = g_movedPatterns[pid];

    double localBandwidth, density = 0.0;

    gsl_vector_view xi;
    determineGlobalKernelShape(x, pid);

    g_kernel.computeConstants(globalBandwidthMatrix, pid);

    for(size_t i = 0; i < g_numXs; i++){
        xi = gsl_matrix_row(g_xs, i);

        //x - xi
        movedPattern = gsl_subtract(x, &xi.vector, movedPattern);
        localBandwidth = gsl_vector_get(g_localBandwidths, i);

        double kernelResult = g_kernel.density(movedPattern, localBandwidth, pid);

        density += kernelResult;
    }

    density /= g_numXs;

    return density;
}

void determineGlobalKernelShape(gsl_vector* x, int pid) {
    gsl_matrix* nearestNeighbours = g_nearestNeighbourMatrices[pid];
    gsl_matrix* globalBandwidthMatrix = g_globalBandwidthMatrices[pid];

    /* Compute K nearest neighbours */
    computeKNearestNeighbours(x, g_k, nearestNeighbours);

    /* Compute the covariance matrix of the neighbours */
    computeCovarianceMatrix(nearestNeighbours, globalBandwidthMatrix);

    /* Compute the scaling factor */
    double scalingFactor = computeScalingFactor(g_globalBandwidthFactor, globalBandwidthMatrix);

    /* Scale the shape matrix */
    gsl_matrix_scale(globalBandwidthMatrix, scalingFactor);
}

void allocateGlobals(size_t dataDimension, size_t num_xi_s, size_t k) {
    g_globalBandwidthMatrices = gsl_matrices_alloc(dataDimension, dataDimension, g_numThreads);
    g_nearestNeighbourMatrices = gsl_matrices_alloc(k, dataDimension, g_numThreads);
    g_movedPatterns = gsl_vectors_alloc(dataDimension, g_numThreads);

    g_kernel.allocate(dataDimension, g_numThreads);
}

void freeGlobals() {
    gsl_matrices_free(g_globalBandwidthMatrices, g_numThreads);
    gsl_matrices_free(g_nearestNeighbourMatrices, g_numThreads);
    gsl_vectors_free(g_movedPatterns, g_numThreads);
    
    g_kernel.free();
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
