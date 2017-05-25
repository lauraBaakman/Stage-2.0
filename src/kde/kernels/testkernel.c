#include <gsl/gsl_vector_double.h>
#include "testkernel.ih"

Kernel testKernel = {
        .isShapeAdaptive = false,
        .kernel.symmetricKernel.density = normal_pdf,
        .kernel.symmetricKernel.free = normal_free,
        .kernel.symmetricKernel.prepare = normal_prepare,
};

static double g_normal_constant;

static double normal_pdf(gsl_vector* pattern){
    double density = 0;
    for ( size_t i = 0; i < pattern->size; i++ ) {
        density += pattern->data[i];
    }
    density = fabs(density * g_normal_constant);

    return density;
}

static void normal_prepare(size_t dimension){
    g_normal_constant = testKernelConstant(dimension);
}

static void normal_free(){
    g_normal_constant = 0.0;
}

double testKernelConstant(int patternDimensionality) {
    return 1.0 / patternDimensionality;
}

double testKernelPDF(gsl_vector* pattern, double constant) {
    double density = 0;
    for ( size_t i = 0; i < pattern->size; i++ ) {
        density += pattern->data[i];
    }
    double mean = density * constant;
    return fabs(mean);
}