#include "parzen.ih"

static SymmetricKernel g_kernel;

static double g_parzenFactor;
static double g_globalBandwidthFactor;

static gsl_matrix* g_xis;

static int g_numThreads;

static gsl_vector** g_scaledPatterns;

void parzen(gsl_matrix *xs, gsl_matrix *xis,
            double windowWidth, KernelType kernelType,
            gsl_vector* densities, gsl_vector* numUsedPatterns) {

    prepareGlobals(xis, windowWidth, kernelType);

    #pragma omp parallel shared(densities, xs)
    {
        int pid = omp_get_thread_num();
        gsl_vector_view x;
        double density;
        int usedPatternCount = 0;

        #pragma omp for
        for(size_t j = 0; j < xs->size1; j++)
        {
            x = gsl_matrix_row(xs, j);
            density = singlePattern(&x.vector, &usedPatternCount, pid);

            gsl_vector_set(numUsedPatterns, j, (double) usedPatternCount);
            gsl_vector_set(densities, j, density);
        }
    }
    freeGlobals();
}

double singlePattern(gsl_vector *x, int* usedPatternCount, int pid){
    gsl_vector* scaledPattern = g_scaledPatterns[pid];

    *usedPatternCount = 0;

    double density = 0;
    double term;

    gsl_vector_view xi;

    for (size_t i = 0; i < g_xis->size1; ++i)
    {
        xi = gsl_matrix_row(g_xis, i);

        gsl_subtract(x, &xi.vector, scaledPattern);
        gsl_vector_scale(scaledPattern, g_globalBandwidthFactor);

        term = g_kernel.density(scaledPattern, 0);
        density += term;

        (*usedPatternCount) += (term > 0.0);
    }
    density *= g_parzenFactor;
    return density;
}

void prepareGlobals(gsl_matrix *xis, double globalBandwidth, KernelType kernelType) {
    g_numThreads = 1;
    #pragma omp parallel 
    { 
        g_numThreads = omp_get_num_threads(); 
    }    
    if(getenv("DEBUGOUTPUT") != NULL){
        printf("\t\t\tnum threads: %d\n", g_numThreads);
    }

    size_t dimension = xis->size2;

    g_xis = xis;

    g_globalBandwidthFactor = computeGlobalBandwidthFactor(globalBandwidth);
    g_parzenFactor = computeParzenFactor(globalBandwidth, xis);


    g_kernel = selectSymmetricKernel(kernelType);
    g_kernel.prepare(dimension, 1);

    allocateGlobals(dimension);
}

double computeParzenFactor(double globalBandwidth, gsl_matrix* xis){
    return 1.0 / (xis->size1 * pow(globalBandwidth, xis->size2));
}

double computeGlobalBandwidthFactor(double globalBandwidth) {
    return 1.0 / globalBandwidth;
}

void allocateGlobals(size_t dataDimension) {
    g_scaledPatterns = gsl_vectors_alloc(dataDimension, g_numThreads);
}

void freeGlobals() {
    g_parzenFactor = 0.0;
    g_globalBandwidthFactor = 0.0;

    gsl_vectors_free(g_scaledPatterns, g_numThreads);

    g_xis = NULL;

    g_kernel.free();
}

