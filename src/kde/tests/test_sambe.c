#include "test_sambe.h"

void testSambe(CuTest* tc){
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

    ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(SHAPE_ADAPTIVE_EPANECHNIKOV);

    gsl_vector* actual = gsl_vector_alloc(xs->size1);

    int k = 3;

    sambe(xs, localBandwidths, globalBandwidth, kernel, k, actual);

    gsl_vector* expected = gsl_vector_alloc(xs->size1);
    gsl_vector_set(expected, 0, 0.567734888282212);
    gsl_vector_set(expected, 1, 0.283867145804328);
    gsl_vector_set(expected, 2, 0.283867145804328);
    gsl_vector_set(expected, 3, 0.567734888282212);

    CuAssertVectorEquals(tc, expected, actual, delta);

    gsl_matrix_free(xs);
    gsl_vector_free(actual);
    gsl_vector_free(expected);
    gsl_vector_free(localBandwidths);
}

CuSuite *SAMBEGetSuite() {
	CuSuite *suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, testSambe);
	return suite;
}