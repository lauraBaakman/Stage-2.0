#include "test_sambe.h"

void testDetermineGlobalKernelShape(CuTest* tc){
    gsl_matrix* xs = gsl_matrix_alloc(4, 2);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 1);
    gsl_matrix_set(xs, 0, 0, 1); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 0, 0, 1); gsl_matrix_set(xs, 0, 1, 1);

    gsl_vector* localBandwidths = gsl_vector_alloc(4);
    gsl_vector_set(localBandwidths, 0, 0.840896194313949);
    gsl_vector_set(localBandwidths, 1, 1.189207427458816);
    gsl_vector_set(localBandwidths, 2, 1.189207427458816);
    gsl_vector_set(localBandwidths, 3, 0.840896194313949);

    double globalBandwidth = 0.721347520444482;

    ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(SHAPE_ADAPTIVE_GAUSSIAN);

    prepareGlobals(xs, localBandwidths, globalBandwidth, kernel);

    size_t pattern_idx = 0;
    determineGlobalKernelShape(pattern_idx);

    gsl_matrix* expected = gsl_matrix_alloc(2, 2);
    gsl_matrix_set(expected, 0, 0, 0.832940370215782); gsl_matrix_set(expected, 0, 1, -0.416470185107891);
    gsl_matrix_set(expected, 1, 1, - 0.416470185107891); gsl_matrix_set(expected, 1, 1, 0.832940370215782);

    gsl_matrix* actual = g_globalKernelShape;

    CuAssertMatrixEquals(tc, expected, actual, delta);
}

CuSuite *SAMBEGetSuite() {
	CuSuite *suite = CuSuiteNew();
	SUITE_ADD_TEST(suite, testDetermineGlobalKernelShape);
	return suite;
}