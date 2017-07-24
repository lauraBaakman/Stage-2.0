#include "test_mbe.h"

#include "../mbe.h"

#include <stdio.h>
#include "omp.h"

#include "../../lib/CuTestUtils.h"
#include "../../test_utils.h"

#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_matrix.h>
#include "../kernels/kernels.h"

void testMBESingleThreaded(CuTest *tc){
	limit_num_threads_to(1);

	size_t numXs = 6;

    gsl_matrix* xs = gsl_matrix_alloc(numXs, 2);
	gsl_matrix_set(xs, 0, 0, +0.00);	gsl_matrix_set(xs, 0, 1, +0.00);
	gsl_matrix_set(xs, 1, 0, +1.00);	gsl_matrix_set(xs, 1, 1, +1.00);
	gsl_matrix_set(xs, 2, 0, +0.00);	gsl_matrix_set(xs, 2, 1, +1.00);
	gsl_matrix_set(xs, 3, 0, +0.50);	gsl_matrix_set(xs, 3, 1, +0.20);
	gsl_matrix_set(xs, 4, 0, +0.30);	gsl_matrix_set(xs, 4, 1, +0.70);
	gsl_matrix_set(xs, 5, 0, -0.23);	gsl_matrix_set(xs, 5, 1, +0.33);

    gsl_matrix* xis = gsl_matrix_alloc(3, 2);
	gsl_matrix_set(xis, 0, 0, -1.00);	gsl_matrix_set(xis, 0, 1, -1.00);
	gsl_matrix_set(xis, 1, 0, +1.00);	gsl_matrix_set(xis, 1, 1, +1.00);
	gsl_matrix_set(xis, 2, 0, +0.00);	gsl_matrix_set(xis, 2, 1, +0.00);

	double globalBandwidth = 0.5;

    KernelType kernelType = STANDARD_GAUSSIAN;

    gsl_vector* actual_densities = gsl_vector_alloc(numXs);
    gsl_vector* actual_num_patterns = gsl_vector_alloc(numXs);

    gsl_vector* expected = gsl_vector_alloc(numXs);
	gsl_vector_set(expected, 0, 0.002648978899632);
	gsl_vector_set(expected, 1, 0.002423568692849);
	gsl_vector_set(expected, 2, 0.002532809909366);
	gsl_vector_set(expected, 3, 0.002584120590522);
	gsl_vector_set(expected, 4, 0.002550145083789);
	gsl_vector_set(expected, 5, 0.002634368311415);

    gsl_vector* localBandwidths = gsl_vector_alloc(xis->size1);
    gsl_vector_set(localBandwidths, 0, 10);
    gsl_vector_set(localBandwidths, 1, 20);
    gsl_vector_set(localBandwidths, 2, 50);

    mbe(xs, xis, 
    	globalBandwidth, localBandwidths, 
    	kernelType, actual_densities, actual_num_patterns);

    CuAssertVectorEquals(tc, expected, actual_densities, delta);

    reset_omp();

    gsl_vector_free(actual_densities);
    gsl_vector_free(actual_num_patterns);
    gsl_vector_free(expected);
    gsl_matrix_free(xs);
    gsl_matrix_free(xis);

	reset_omp();
}

void testMBEMultiThreaded(CuTest *tc){
	limit_num_threads_to(2);

	size_t numXs = 6;

    gsl_matrix* xs = gsl_matrix_alloc(numXs, 2);
	gsl_matrix_set(xs, 0, 0, +0.00);	gsl_matrix_set(xs, 0, 1, +0.00);
	gsl_matrix_set(xs, 1, 0, +1.00);	gsl_matrix_set(xs, 1, 1, +1.00);
	gsl_matrix_set(xs, 2, 0, +0.00);	gsl_matrix_set(xs, 2, 1, +1.00);
	gsl_matrix_set(xs, 3, 0, +0.50);	gsl_matrix_set(xs, 3, 1, +0.20);
	gsl_matrix_set(xs, 4, 0, +0.30);	gsl_matrix_set(xs, 4, 1, +0.70);
	gsl_matrix_set(xs, 5, 0, -0.23);	gsl_matrix_set(xs, 5, 1, +0.33);

    gsl_matrix* xis = gsl_matrix_alloc(3, 2);
	gsl_matrix_set(xis, 0, 0, -1.00);	gsl_matrix_set(xis, 0, 1, -1.00);
	gsl_matrix_set(xis, 1, 0, +1.00);	gsl_matrix_set(xis, 1, 1, +1.00);
	gsl_matrix_set(xis, 2, 0, +0.00);	gsl_matrix_set(xis, 2, 1, +0.00);

	double globalBandwidth = 0.5;

    KernelType kernelType = STANDARD_GAUSSIAN;

    gsl_vector* actual_densities = gsl_vector_alloc(numXs);
    gsl_vector* actual_num_patterns = gsl_vector_alloc(numXs);

    gsl_vector* expected = gsl_vector_alloc(numXs);
	gsl_vector_set(expected, 0, 0.002648978899632);
	gsl_vector_set(expected, 1, 0.002423568692849);
	gsl_vector_set(expected, 2, 0.002532809909366);
	gsl_vector_set(expected, 3, 0.002584120590522);
	gsl_vector_set(expected, 4, 0.002550145083789);
	gsl_vector_set(expected, 5, 0.002634368311415);

    gsl_vector* localBandwidths = gsl_vector_alloc(xis->size1);
    gsl_vector_set(localBandwidths, 0, 10);
    gsl_vector_set(localBandwidths, 1, 20);
    gsl_vector_set(localBandwidths, 2, 50);

    mbe(xs, xis, 
        globalBandwidth, localBandwidths, 
        kernelType, actual_densities, actual_num_patterns);

    CuAssertVectorEquals(tc, expected, actual_densities, delta);

    reset_omp();

    gsl_vector_free(actual_densities);
    gsl_vector_free(actual_num_patterns);
    gsl_vector_free(expected);
    gsl_matrix_free(xs);
    gsl_matrix_free(xis);

	reset_omp();
}

CuSuite *MBEGetSuite() {
    CuSuite *suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, testMBESingleThreaded);
    SUITE_ADD_TEST(suite, testMBEMultiThreaded);
    return suite;
}