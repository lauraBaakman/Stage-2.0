#include "test_parzen.h"

#include "../sambe.h"

#include <stdio.h>
#include "omp.h"

#include "../../lib/CuTestUtils.h"
#include "../../test_utils.h"

#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_matrix.h>
#include "../kernels/kernels.h"

void testSAMBESingleThreaded(CuTest *tc){
    size_t numXs = 4;

    int k = 3;

    limit_num_threads_to(1);

    gsl_matrix* xs = gsl_matrix_alloc(numXs, 2);
	gsl_matrix_set(xs, 0, 0, +0.00);	gsl_matrix_set(xs, 0, 1, +0.00);
	gsl_matrix_set(xs, 1, 0, +0.00);	gsl_matrix_set(xs, 1, 1, +1.00);
	gsl_matrix_set(xs, 2, 0, +1.00);	gsl_matrix_set(xs, 2, 1, +0.00);
	gsl_matrix_set(xs, 3, 0, +1.00);	gsl_matrix_set(xs, 3, 1, +1.00);

    double globalBandwidth = 0.721347520444482;

    KernelType kernelType = SHAPE_ADAPTIVE_GAUSSIAN;

    gsl_vector* actual = gsl_vector_alloc(numXs);

    gsl_vector* expected = gsl_vector_alloc(numXs);
    gsl_vector_set(expected, 0, 0.186693239491116);
    gsl_vector_set(expected, 1, 0.077446155260620);
    gsl_vector_set(expected, 2, 0.077446155260620);
    gsl_vector_set(expected, 3, 0.143018801263046);

    gsl_vector* localBandwidths = gsl_vector_alloc(xs->size1);
    gsl_vector_set(localBandwidths, 0, 0.840896194314);
    gsl_vector_set(localBandwidths, 1, 1.18920742746);
    gsl_vector_set(localBandwidths, 2, 1.18920742746);
    gsl_vector_set(localBandwidths, 3, 0.840896194314);

    gsl_matrix* xis = xs;

    sambe(xs, xis,
    	localBandwidths, globalBandwidth,
    	kernelType, k,
    	actual);

    CuAssertVectorEquals(tc, expected, actual, delta);

    reset_omp();

    gsl_vector_free(actual);
    gsl_vector_free(expected);
    gsl_matrix_free(xs);
    gsl_vector_free(localBandwidths);
}

void testSAMBEMultiThreaded(CuTest *tc){
    size_t numXs = 4;

    int k = 3;

    limit_num_threads_to(2);

    gsl_matrix* xs = gsl_matrix_alloc(numXs, 2);
	gsl_matrix_set(xs, 0, 0, +0.00);	gsl_matrix_set(xs, 0, 1, +0.00);
	gsl_matrix_set(xs, 1, 0, +0.00);	gsl_matrix_set(xs, 1, 1, +1.00);
	gsl_matrix_set(xs, 2, 0, +1.00);	gsl_matrix_set(xs, 2, 1, +0.00);
	gsl_matrix_set(xs, 3, 0, +1.00);	gsl_matrix_set(xs, 3, 1, +1.00);

    double globalBandwidth = 0.721347520444482;

    KernelType kernelType = SHAPE_ADAPTIVE_GAUSSIAN;

    gsl_vector* actual = gsl_vector_alloc(numXs);

    gsl_vector* expected = gsl_vector_alloc(numXs);
    gsl_vector_set(expected, 0, 0.186693239491116);
    gsl_vector_set(expected, 1, 0.077446155260620);
    gsl_vector_set(expected, 2, 0.077446155260620);
    gsl_vector_set(expected, 3, 0.143018801263046);

    gsl_vector* localBandwidths = gsl_vector_alloc(xs->size1);
    gsl_vector_set(localBandwidths, 0, 0.840896194314);
    gsl_vector_set(localBandwidths, 1, 1.18920742746);
    gsl_vector_set(localBandwidths, 2, 1.18920742746);
    gsl_vector_set(localBandwidths, 3, 0.840896194314);

    gsl_matrix* xis = xs;

    sambe(xs, xis,
    	localBandwidths, globalBandwidth,
    	kernelType, k,
    	actual);

    CuAssertVectorEquals(tc, expected, actual, delta);

    reset_omp();

    gsl_vector_free(actual);
    gsl_vector_free(expected);
    gsl_matrix_free(xs);
    gsl_vector_free(localBandwidths);
}

CuSuite *SAMBEGetSuite() {
    CuSuite *suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, testSAMBESingleThreaded);
    SUITE_ADD_TEST(suite, testSAMBEMultiThreaded);
    return suite;
}