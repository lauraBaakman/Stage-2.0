#include "test_mbe.h"

void testMBE(CuTest* tc){
    gsl_matrix* xis = gsl_matrix_alloc(3, 2);
    gsl_matrix_set(xis, 0, 0, -1); gsl_matrix_set(xis, 0, 1, -1);
    gsl_matrix_set(xis, 1, 0, +1); gsl_matrix_set(xis, 1, 1, +1);
    gsl_matrix_set(xis, 2, 0, +0); gsl_matrix_set(xis, 2, 1, +0);

    gsl_matrix* xs = gsl_matrix_alloc(3, 2);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 1, 0, 1); gsl_matrix_set(xs, 1, 1, 1);
    gsl_matrix_set(xs, 2, 0, 0); gsl_matrix_set(xs, 2, 1, 1);

    gsl_vector* localBandwidths = gsl_vector_alloc(3);
    gsl_vector_set(localBandwidths, 0, 10);
    gsl_vector_set(localBandwidths, 1, 20);
    gsl_vector_set(localBandwidths, 2, 50);

    double globalBandwidth = 0.5;

    KernelType kernel = EPANECHNIKOV;

    gsl_vector* actual = gsl_vector_alloc(xs->size1);

    mbe(xs, xis, globalBandwidth, localBandwidths, kernel, actual);

    gsl_vector* expected = gsl_vector_alloc(xs->size1);
    gsl_vector_set(expected, 0, 0.013128790727490);
    gsl_vector_set(expected, 1, 0.009690664372166);
    gsl_vector_set(expected, 2, 0.011409727549828);

    CuAssertVectorEquals(tc, expected, actual, delta);

    gsl_matrix_free(xs);
    gsl_matrix_free(xis);
    gsl_vector_free(actual);
    gsl_vector_free(expected);
    gsl_vector_free(localBandwidths);
}

CuSuite *MBEGetSuite() {
	CuSuite *suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, testMBE);
	return suite;
}