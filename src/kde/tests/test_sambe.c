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
    size_t numXs = 5;
    size_t numXis = 4;

    int k = 3;

    limit_num_threads_to(1);

    gsl_matrix* xs = gsl_matrix_alloc(numXs, 2);
	gsl_matrix_set(xs, 0, 0, +0.00);	gsl_matrix_set(xs, 0, 1, +0.00);
	gsl_matrix_set(xs, 1, 0, +0.00);	gsl_matrix_set(xs, 1, 1, +1.00);
	gsl_matrix_set(xs, 2, 0, +1.00);	gsl_matrix_set(xs, 2, 1, +0.00);
	gsl_matrix_set(xs, 3, 0, +1.00);	gsl_matrix_set(xs, 3, 1, +1.00);
    gsl_matrix_set(xs, 4, 0, +2.00);    gsl_matrix_set(xs, 4, 1, +2.00);    

    gsl_matrix* xis = gsl_matrix_alloc(numXis, 2);
    gsl_matrix_set(xis, 0, 0, +0.00);    gsl_matrix_set(xis, 0, 1, +0.00);
    gsl_matrix_set(xis, 1, 0, +0.00);    gsl_matrix_set(xis, 1, 1, +1.00);
    gsl_matrix_set(xis, 2, 0, +1.00);    gsl_matrix_set(xis, 2, 1, +0.00);
    gsl_matrix_set(xis, 3, 0, +1.00);    gsl_matrix_set(xis, 3, 1, +1.00);

    double globalBandwidth = 0.721347520444482;

    KernelType kernelType = SHAPE_ADAPTIVE_GAUSSIAN;

    gsl_vector* actual_densities = gsl_vector_alloc(numXs);
    gsl_vector* actual_pattern_count = gsl_vector_alloc(numXs);
    gsl_matrix* actual_eigen_values = gsl_matrix_alloc(4, 2);
    gsl_matrix* actual_eigen_vectors = gsl_matrix_alloc(4, 4);

    gsl_vector* expected = gsl_vector_alloc(numXs);
    gsl_vector_set(expected, 0, 0.143018801266957);
    gsl_vector_set(expected, 1, 0.077446155261498);
    gsl_vector_set(expected, 2, 0.077446155261498);
    gsl_vector_set(expected, 3, 0.186693239495190);
    gsl_vector_set(expected, 4, 0.017000356330535);    

    gsl_vector* localBandwidths = gsl_vector_alloc(numXis);
    gsl_vector_set(localBandwidths, 0, 0.840896194314);
    gsl_vector_set(localBandwidths, 1, 1.18920742746);
    gsl_vector_set(localBandwidths, 2, 1.18920742746);
    gsl_vector_set(localBandwidths, 3, 0.840896194314);

    sambe(xs, xis,
    	localBandwidths, globalBandwidth,
    	kernelType, k,
    	actual_densities, actual_pattern_count, 
        actual_eigen_values, actual_eigen_vectors);

    CuAssertVectorEquals(tc, expected, actual_densities, delta);

    reset_omp();

    gsl_vector_free(actual_densities);
    gsl_vector_free(actual_pattern_count);
    gsl_vector_free(expected);
    gsl_matrix_free(xs);
    gsl_matrix_free(xis);
    gsl_vector_free(localBandwidths);
    gsl_matrix_free(actual_eigen_values);
    gsl_matrix_free(actual_eigen_vectors);
}

void testSAMBEMultiThreaded(CuTest *tc){
    size_t numXs = 5;
    size_t numXis = 4;

    int k = 3;

    limit_num_threads_to(2);

    gsl_matrix* xs = gsl_matrix_alloc(numXs, 2);
    gsl_matrix_set(xs, 0, 0, +0.00);    gsl_matrix_set(xs, 0, 1, +0.00);
    gsl_matrix_set(xs, 1, 0, +0.00);    gsl_matrix_set(xs, 1, 1, +1.00);
    gsl_matrix_set(xs, 2, 0, +1.00);    gsl_matrix_set(xs, 2, 1, +0.00);
    gsl_matrix_set(xs, 3, 0, +1.00);    gsl_matrix_set(xs, 3, 1, +1.00);
    gsl_matrix_set(xs, 4, 0, +2.00);    gsl_matrix_set(xs, 4, 1, +2.00);    

    gsl_matrix* xis = gsl_matrix_alloc(numXis, 2);
    gsl_matrix_set(xis, 0, 0, +0.00);    gsl_matrix_set(xis, 0, 1, +0.00);
    gsl_matrix_set(xis, 1, 0, +0.00);    gsl_matrix_set(xis, 1, 1, +1.00);
    gsl_matrix_set(xis, 2, 0, +1.00);    gsl_matrix_set(xis, 2, 1, +0.00);
    gsl_matrix_set(xis, 3, 0, +1.00);    gsl_matrix_set(xis, 3, 1, +1.00);

    double globalBandwidth = 0.721347520444482;

    KernelType kernelType = SHAPE_ADAPTIVE_GAUSSIAN;

    gsl_vector* actual_densities = gsl_vector_alloc(numXs);
    gsl_vector* actual_pattern_count = gsl_vector_alloc(numXs);
    gsl_matrix* actual_eigen_values = gsl_matrix_alloc(4, 2);
    gsl_matrix* actual_eigen_vectors = gsl_matrix_alloc(4, 4);    

    gsl_vector* expected = gsl_vector_alloc(numXs);
    gsl_vector_set(expected, 0, 0.143018801266957);
    gsl_vector_set(expected, 1, 0.077446155261498);
    gsl_vector_set(expected, 2, 0.077446155261498);
    gsl_vector_set(expected, 3, 0.186693239495190);
    gsl_vector_set(expected, 4, 0.017000356330535);    

    gsl_vector* localBandwidths = gsl_vector_alloc(numXis);
    gsl_vector_set(localBandwidths, 0, 0.840896194314);
    gsl_vector_set(localBandwidths, 1, 1.18920742746);
    gsl_vector_set(localBandwidths, 2, 1.18920742746);
    gsl_vector_set(localBandwidths, 3, 0.840896194314);

    sambe(xs, xis,
        localBandwidths, globalBandwidth,
        kernelType, k,
        actual_densities, actual_pattern_count,
        actual_eigen_values, actual_eigen_vectors);

    CuAssertVectorEquals(tc, expected, actual_densities, delta);

    reset_omp();

    gsl_vector_free(actual_densities);
    gsl_vector_free(actual_pattern_count);
    gsl_vector_free(expected);
    gsl_matrix_free(xs);
    gsl_matrix_free(xis);
    gsl_vector_free(localBandwidths);
    gsl_matrix_free(actual_eigen_values);
    gsl_matrix_free(actual_eigen_vectors);

}

CuSuite *SAMBEGetSuite() {
    CuSuite *suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, testSAMBESingleThreaded);
    SUITE_ADD_TEST(suite, testSAMBEMultiThreaded);
    return suite;
}