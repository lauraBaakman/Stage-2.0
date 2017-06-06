#include "test_sambe.h"

void testDetermineGlobalKernelShape(CuTest* tc){
    gsl_matrix* xs = gsl_matrix_alloc(4, 2);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 1, 0, 0); gsl_matrix_set(xs, 1, 1, 1);
    gsl_matrix_set(xs, 2, 0, 1); gsl_matrix_set(xs, 2, 1, 0);
    gsl_matrix_set(xs, 3, 0, 1); gsl_matrix_set(xs, 3, 1, 1);

    gsl_vector* localBandwidths = gsl_vector_alloc(4);
    gsl_vector_set(localBandwidths, 0, 0.840896194313949);
    gsl_vector_set(localBandwidths, 1, 1.189207427458816);
    gsl_vector_set(localBandwidths, 2, 1.189207427458816);
    gsl_vector_set(localBandwidths, 3, 0.840896194313949);

    double globalBandwidth = 0.721347520444482;

    ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(SHAPE_ADAPTIVE_GAUSSIAN);

    int k = 3;

    prepareGlobals(xs, localBandwidths, globalBandwidth, kernel, k);

    size_t pattern_idx = 0;
    determineGlobalKernelShape(pattern_idx);

    gsl_matrix* expected = gsl_matrix_alloc(2, 2);
    gsl_matrix_set(expected, 0, 0, 0.832940370215782); gsl_matrix_set(expected, 0, 1, -0.416470185107891);
    gsl_matrix_set(expected, 1, 0, - 0.416470185107891); gsl_matrix_set(expected, 1, 1, 0.832940370215782);

    gsl_matrix* actual = g_globalBandwidthMatrix;

    CuAssertMatrixEquals(tc, expected, actual, delta);

    gsl_matrix_free(xs);
    gsl_vector_free(localBandwidths);
    gsl_matrix_free(expected);
    freeGlobals();

}

void testFinalDensitySinglePattern(CuTest* tc){
    gsl_matrix* xs = gsl_matrix_alloc(4, 2);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 1, 0, 0); gsl_matrix_set(xs, 1, 1, 1);
    gsl_matrix_set(xs, 2, 0, 1); gsl_matrix_set(xs, 2, 1, 0);
    gsl_matrix_set(xs, 3, 0, 1); gsl_matrix_set(xs, 3, 1, 1);

    gsl_vector* localBandwidths = gsl_vector_alloc(4);
    gsl_vector_set(localBandwidths, 0, 0.840896194313949);
    gsl_vector_set(localBandwidths, 1, 1.189207427458816);
    gsl_vector_set(localBandwidths, 2, 1.189207427458816);
    gsl_vector_set(localBandwidths, 3, 0.840896194313949);

    double globalBandwidth = 0.721347520444482;

    ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(SHAPE_ADAPTIVE_GAUSSIAN);

    int k = 3;
    size_t dimension = 2;

    prepareGlobals(xs, localBandwidths, globalBandwidth, kernel, k);
    kernel.allocate(dimension);

    size_t xIdx = 0;
    gsl_vector_view x = gsl_matrix_row(xs, xIdx);

    double actual = finalDensitySinglePattern(&x.vector, xIdx);

    double expected = 0.143018801263046;

    CuAssertDblEquals(tc, expected, actual, delta);

    gsl_matrix_free(xs);
    gsl_vector_free(localBandwidths);
    freeGlobals();
    kernel.free();
}

CuSuite *SAMBEGetSuite() {
	CuSuite *suite = CuSuiteNew();
	SUITE_ADD_TEST(suite, testDetermineGlobalKernelShape);
    SUITE_ADD_TEST(suite, testFinalDensitySinglePattern);
	return suite;
}