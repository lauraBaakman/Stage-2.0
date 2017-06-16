#include "test_mbe.h"
#include <omp.h>

void testMBE1(CuTest* tc){
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

void testMBE2(CuTest* tc){
    size_t numXsSmall = 3;
    size_t numXs = 900;
    
    size_t numXis = 3;
    size_t dimension = 2;

    gsl_matrix* xis = gsl_matrix_alloc(numXis, dimension);
    gsl_matrix_set(xis, 0, 0, -1); gsl_matrix_set(xis, 0, 1, -1);
    gsl_matrix_set(xis, 1, 0, +1); gsl_matrix_set(xis, 1, 1, +1);
    gsl_matrix_set(xis, 2, 0, +0); gsl_matrix_set(xis, 2, 1, +0);

    gsl_matrix* xsSmall = gsl_matrix_alloc(numXsSmall, dimension);
    gsl_matrix_set(xsSmall, 0, 0, 0); gsl_matrix_set(xsSmall, 0, 1, 0);
    gsl_matrix_set(xsSmall, 1, 0, 1); gsl_matrix_set(xsSmall, 1, 1, 1);
    gsl_matrix_set(xsSmall, 2, 0, 0); gsl_matrix_set(xsSmall, 2, 1, 1);

    gsl_vector* localBandwidthsSmall = gsl_vector_alloc(numXsSmall);
    gsl_vector_set(localBandwidthsSmall, 0, 10);
    gsl_vector_set(localBandwidthsSmall, 1, 20);
    gsl_vector_set(localBandwidthsSmall, 2, 50);

    gsl_vector* expectedSmall = gsl_vector_alloc(numXsSmall);
    gsl_vector_set(expectedSmall, 0, 0.013128790727490);
    gsl_vector_set(expectedSmall, 1, 0.009690664372166);
    gsl_vector_set(expectedSmall, 2, 0.011409727549828);

    gsl_matrix* xs = gsl_matrix_alloc(numXs, dimension);
    gsl_vector* localBandwidths = gsl_vector_alloc(numXs);
    gsl_vector* expected = gsl_vector_alloc(numXs);
    for(size_t i = 0, smallI = 0; i < numXs; i++, smallI++){
        smallI %= numXsSmall;
        gsl_matrix_set(xs, i, 0, gsl_matrix_get(xsSmall, smallI, 0)); 
        gsl_matrix_set(xs, i, 1, gsl_matrix_get(xsSmall, smallI, 1));

        gsl_vector_set(localBandwidths, i, gsl_vector_get(localBandwidthsSmall, smallI));
        gsl_vector_set(expected, i, gsl_vector_get(expectedSmall, smallI));
    }

    double globalBandwidth = 0.5;

    KernelType kernel = EPANECHNIKOV;

    gsl_vector* actual = gsl_vector_alloc(numXs);

    mbe(xs, xis, globalBandwidth, localBandwidths, kernel, actual);

    CuAssertVectorEquals(tc, expected, actual, delta);

    gsl_matrix_free(xs);
    gsl_matrix_free(xis);
    gsl_vector_free(actual);
    gsl_vector_free(expected);
    gsl_vector_free(localBandwidths);
    gsl_matrix_free(xsSmall);
    gsl_vector_free(localBandwidthsSmall);
    gsl_vector_free(expectedSmall);
}

CuSuite *MBEGetSuite() {
	CuSuite *suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, testMBE1);
    SUITE_ADD_TEST(suite, testMBE2);
	return suite;
}