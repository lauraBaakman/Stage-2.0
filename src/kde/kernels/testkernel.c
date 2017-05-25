#include "testkernel.ih"

Kernel testKernel = {
        .isShapeAdaptive = false,
        .kernel.symmetricKernel.densityFunction = testKernelPDF,
        .kernel.symmetricKernel.factorFunction = testKernelConstant,
};

static double normal_pdf(gsl_vector* pattern){

}

static void normal_prepare(size_t dimension){

}

static void normal_free(){

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