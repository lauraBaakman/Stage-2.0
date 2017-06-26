#include "mbe.ih"

static SymmetricKernel g_kernel;

static gsl_matrix* g_xis;

static double g_globalBandwidth;
static gsl_vector* g_localBandwidths;

static gsl_vector** g_scaledPatterns;

static int g_numThreads;

void mbe(gsl_matrix *xs, gsl_matrix *xis,
         double globalBandwidth, gsl_vector *localBandwidths,
         KernelType kernelType, gsl_vector *densities) {

    prepareGlobals(xis, globalBandwidth, localBandwidths, kernelType);

    printf("MBE num threads: %d\n", g_numThreads);

    #pragma omp parallel shared(xs, densities)
    {
        double density;
        int pid = omp_get_thread_num();
        gsl_vector_view x;

        #pragma omp for
        for(size_t j = 0; j < xs->size1; j++)
        {
            x = gsl_matrix_row(xs, j);
            density = estimateSinglePattern(&x.vector, pid);
            gsl_vector_set(densities, j, density);
        }
    }

    freeGlobals();
}

void prepareGlobals(gsl_matrix *xis, double globalBandwidth, gsl_vector *localBandwidths, KernelType kernelType) {

    g_numThreads = 1;
    #pragma omp parallel 
    {
        g_numThreads = omp_get_num_threads();
    }

    g_kernel = selectSymmetricKernel(kernelType);
    g_kernel.prepare(xis->size2, g_numThreads);

    g_xis = xis;

    g_globalBandwidth = globalBandwidth;
    g_localBandwidths = localBandwidths;

    allocateGlobals(xis->size2);
}

void allocateGlobals(size_t dataDimension){
    g_scaledPatterns = gsl_vectors_alloc(dataDimension, g_numThreads);
}

double estimateSinglePattern(gsl_vector *x, int pid) {
    gsl_vector_view xi;

    double density = 0;
    double factor, bandwidth;

    gsl_vector* scaledPattern = g_scaledPatterns[pid];

    for(size_t i = 0; i < g_xis->size1; ++i){
        xi = gsl_matrix_row(g_xis, i);

        bandwidth = computeBandwidth(gsl_vector_get(g_localBandwidths, i));
        factor = computeFactor(bandwidth, g_xis->size2);

        scale(x, &xi.vector, scaledPattern, bandwidth);

        density += (factor * g_kernel.density(scaledPattern, pid));
    }
    density /= (double) g_xis->size1;

    return density;
}

double computeBandwidth(double localBandwidth) {
    return g_globalBandwidth * localBandwidth;
}

double computeFactor(double bandwidth, size_t dimension) {
    return pow(bandwidth, -1 * (int) dimension);
}

void scale(gsl_vector *x, gsl_vector *xi, gsl_vector *result, double bandwidth) {
    gsl_subtract(x, xi, result);
    gsl_vector_scale(result, 1.0 / bandwidth);
}

void freeGlobals(){
    g_kernel.free();

    g_xis = NULL;

    g_globalBandwidth = 0.0;
    g_localBandwidths = NULL;

    gsl_vectors_free(g_scaledPatterns, g_numThreads);
}
