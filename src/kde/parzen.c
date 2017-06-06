#include "parzen.ih"

static SymmetricKernel g_kernel;

static double g_parzenFactor;
static double g_globalBandwidthFactor;

static gsl_matrix* g_xis;

static gsl_vector* g_scaledPattern;

void parzen(gsl_matrix *xs, gsl_matrix *xis,
            double windowWidth, SymmetricKernel kernel,
            gsl_vector* outDensities) {

    prepareGlobals(xis, windowWidth, kernel);

    gsl_vector_view x;

    for(size_t j = 0; j < xs->size1; j++)
    {
        x = gsl_matrix_row(xs, j);
        double density = singlePattern(&x.vector);
        gsl_vector_set(outDensities, j, density);
    }
    freeGlobals();
}

double singlePattern(gsl_vector *x){
    double density = 0;

    gsl_vector_view xi;

    for (size_t i = 0; i < g_xis->size1; ++i)
    {
        xi = gsl_matrix_row(g_xis, i);

        gsl_subtract(x, &xi.vector, g_scaledPattern);
        gsl_vector_scale(g_scaledPattern, g_globalBandwidthFactor);

        density += g_kernel.density(g_scaledPattern);
    }
    density *= g_parzenFactor;

    return density;
}

void prepareGlobals(gsl_matrix *xis, double globalBandwidth, SymmetricKernel kernel) {
    size_t dimension = xis->size2;

    g_xis = xis;

    g_globalBandwidthFactor = computeGlobalBandwidthFactor(globalBandwidth);
    g_parzenFactor = computeParzenFactor(globalBandwidth, xis);

    g_kernel = kernel;
    g_kernel.prepare(dimension);

    allocateGlobals(dimension);
}

double computeParzenFactor(double globalBandwidth, gsl_matrix* xis){
    return 1.0 / (xis->size1 * pow(globalBandwidth, xis->size2));
}

double computeGlobalBandwidthFactor(double globalBandwidth) {
    return 1.0 / globalBandwidth;
}

void allocateGlobals(size_t dataDimension) {
    g_scaledPattern = gsl_vector_alloc(dataDimension);
}

void freeGlobals() {
    g_parzenFactor = 0.0;
    g_globalBandwidthFactor = 0.0;

    gsl_vector_free(g_scaledPattern);

    g_xis = NULL;

    g_kernel.free();
}

