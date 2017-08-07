#include "sambe.ih"

gsl_matrix* g_xs;
size_t g_numXs;

gsl_matrix* g_xis;
size_t g_numXis;

gsl_vector* g_localBandwidths;
double g_globalBandwidthFactor;
gsl_matrix** g_globalBandwidthMatrices;

ShapeAdaptiveKernel g_kernel;

gsl_vector** g_movedPatterns;

size_t g_k;
gsl_matrix** g_nearestNeighbourMatrices;

gsl_matrix* g_kernelTerms;

static int g_numThreads;

void sambe(gsl_matrix *xs,
           gsl_matrix *xis,
           gsl_vector *localBandwidths, double globalBandwidth,
           KernelType kernelType, int k,
           gsl_vector *densities, gsl_vector *numUsedPatterns){

    prepareGlobals(xs, xis, localBandwidths, globalBandwidth, kernelType, k);    

    computeKernelTerms(numUsedPatterns);

    gsl_matrix_compute_row_means(g_kernelTerms, densities);

    countNumUsedPatterns(g_kernelTerms, numUsedPatterns);

    freeGlobals();
}

void computeKernelTerms(gsl_vector* numUsedPatterns){
    #pragma omp parallel shared(g_numXs, g_xs, g_xis)
    {
        int pid = omp_get_thread_num();
        gsl_vector_view xi;
        double localBandwidth;

        #pragma omp for
        for(size_t i = 0; i < g_numXis; i++){
            xi = gsl_matrix_row(g_xis, i);
            localBandwidth = gsl_vector_get(g_localBandwidths, i);
            gsl_vector_view terms = gsl_matrix_column(g_kernelTerms, i);

            computeKernelTermxForXi(&xi.vector, localBandwidth, &terms.vector, pid);
        }        
    }   
}

void computeKernelTermxForXi(
    gsl_vector *xi, double localBandwidth,
    gsl_vector* terms, int pid) 
{
    gsl_matrix* globalBandwidthMatrix = g_globalBandwidthMatrices[pid];
    gsl_vector* movedPattern = g_movedPatterns[pid];

    gsl_vector_view x;

    for(size_t i = 0; i < g_numXs; i++){
        determineKernelShape(localBandwidth, xi, pid);
        g_kernel.computeConstants(globalBandwidthMatrix, pid);

        x = gsl_matrix_row(g_xs, i);

        //x - xi
        movedPattern = gsl_subtract(&x.vector, xi, movedPattern);

        double kernelResult = g_kernel.density(movedPattern, pid);
        gsl_vector_set(terms, i, kernelResult);
    }
}

void determineKernelShape(double localBandwidth, gsl_vector* x, int pid) {
    gsl_matrix* nearestNeighbours = g_nearestNeighbourMatrices[pid];
    gsl_matrix* globalBandwidthMatrix = g_globalBandwidthMatrices[pid];

    /* Compute K nearest neighbours */
    computeKNearestNeighbours(x, g_k, nearestNeighbours);

    /* Compute the covariance matrix of the neighbours */
    computeCovarianceMatrix(nearestNeighbours, globalBandwidthMatrix);

    /* Compute the scaling factor */
    double scalingFactor = computeScalingFactor(localBandwidth, g_globalBandwidthFactor, globalBandwidthMatrix);

    /* Scale the shape matrix */
    gsl_matrix_scale(globalBandwidthMatrix, scalingFactor);
}

void countNumUsedPatterns(gsl_matrix* kernelTerms, gsl_vector* numUsedPatterns){
    #pragma omp parallel shared(kernelTerms, numUsedPatterns)
    {
        int usedPatternCount = 0;
        gsl_vector_view row;

        #pragma omp for
        for(size_t i = 0; i < kernelTerms->size1; i++){
            row = gsl_matrix_row(kernelTerms, i);
            usedPatternCount = countValuesGreaterThanZero(&row.vector);
            gsl_vector_set(numUsedPatterns, i, usedPatternCount);
        }
    }
}

int countValuesGreaterThanZero(gsl_vector* vector){
    int count = 0;
    double term;
    for(size_t i = 0; i < vector->size; i++){
        term = gsl_vector_get(vector, i);
        count += (term > 0.0);
    }
    return count;
}

void allocateGlobals(size_t dataDimension, size_t num_xi_s, size_t num_x_s, size_t k) {
    g_globalBandwidthMatrices = gsl_matrices_alloc(dataDimension, dataDimension, g_numThreads);
    g_nearestNeighbourMatrices = gsl_matrices_alloc(k, dataDimension, g_numThreads);
    g_movedPatterns = gsl_vectors_alloc(dataDimension, g_numThreads);
    g_kernelTerms = gsl_matrix_alloc(num_x_s, num_xi_s);

    g_kernel.allocate(dataDimension, g_numThreads);
}

void freeGlobals() {
    gsl_matrices_free(g_globalBandwidthMatrices, g_numThreads);
    gsl_matrices_free(g_nearestNeighbourMatrices, g_numThreads);
    gsl_vectors_free(g_movedPatterns, g_numThreads);
    
    gsl_matrix_free(g_kernelTerms);

    g_kernel.free();
    nn_free();
}

void prepareGlobals(gsl_matrix *xs, gsl_matrix *xis,
                    gsl_vector *localBandwidths, double globalBandwidth,
                    KernelType kernelType, int k) {
    g_numThreads = 1;
    #pragma omp parallel 
    {
        g_numThreads = omp_get_num_threads();
    }
    if(getenv("DEBUGOUTPUT") != NULL){
        printf("\t\t\tnum threads: %d\n", g_numThreads);
    }

    g_kernel = selectShapeAdaptiveKernel(kernelType);

    g_xs = xs;
    g_numXs = xs->size1;

    g_xis = xis;
    g_numXis = xis->size1;

    g_localBandwidths = localBandwidths;
    g_globalBandwidthFactor = globalBandwidth;
    g_k = (size_t) k;

    nn_prepare(xis);
    
    size_t dimension = g_xs->size2;
    allocateGlobals(dimension, g_numXis, g_numXs, g_k);
}