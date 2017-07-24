#include "test_parzen.h"

#include "../parzen.h"

#include <stdio.h>
#include "omp.h"

#include "../../lib/CuTestUtils.h"
#include "../../test_utils.h"

#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_matrix.h>
#include "../kernels/kernels.h"

void testParzenSingleThreaded(CuTest *tc){
    size_t numXs = 6;

    limit_num_threads_to(1);

    gsl_matrix* xs = gsl_matrix_alloc(numXs, 2);
    gsl_matrix_set(xs, 0, 0, 0.00); gsl_matrix_set(xs, 0, 1, 0.0);
    gsl_matrix_set(xs, 1, 0, 0.25); gsl_matrix_set(xs, 1, 1, 0.5);
    gsl_matrix_set(xs, 2, 0, 0.03); gsl_matrix_set(xs, 2, 1, 0.4);
    gsl_matrix_set(xs, 3, 0, 0.72); gsl_matrix_set(xs, 3, 1, 0.1);
    gsl_matrix_set(xs, 4, 0, 0.33); gsl_matrix_set(xs, 4, 1, 0.2);
    gsl_matrix_set(xs, 5, 0, 0.24); gsl_matrix_set(xs, 5, 1, 0.3);

    gsl_matrix* xis = gsl_matrix_alloc(3, 2);
    gsl_matrix_set(xis, 0, 0, -1.00); gsl_matrix_set(xis, 0, 1, -1.0);
    gsl_matrix_set(xis, 1, 0, +0.00); gsl_matrix_set(xis, 1, 1, +0.0);
    gsl_matrix_set(xis, 2, 0, +0.50); gsl_matrix_set(xis, 2, 1, +0.5);

    double globalBandwidth = 4.0;

    KernelType kernelType = STANDARD_GAUSSIAN;

    gsl_vector* actual = gsl_vector_alloc(numXs);
    gsl_vector* actualNumUsedPatterns = gsl_vector_alloc(numXs);

    gsl_vector* expected = gsl_vector_alloc(numXs);
    gsl_vector_set(expected, 0, 0.009694888542698);
    gsl_vector_set(expected, 1, 0.009536078921834);
    gsl_vector_set(expected, 2, 0.009608020510205);
    gsl_vector_set(expected, 3, 0.009466392462365);
    gsl_vector_set(expected, 4, 0.009603126595478);
    gsl_vector_set(expected, 5, 0.009602675165108);

    parzen(xs, xis, globalBandwidth, kernelType, 
        actual, actualNumUsedPatterns);

    CuAssertVectorEquals(tc, expected, actual, delta);

    reset_omp();

    gsl_vector_free(actual);
    gsl_vector_free(expected);
    gsl_matrix_free(xs);
    gsl_matrix_free(xis);
}

void testParzenMultiThreaded(CuTest *tc){
    size_t numXs = 6;

    limit_num_threads_to(2); 

    gsl_matrix* xs = gsl_matrix_alloc(numXs, 2);
    gsl_matrix_set(xs, 0, 0, 0.00); gsl_matrix_set(xs, 0, 1, 0.0);
    gsl_matrix_set(xs, 1, 0, 0.25); gsl_matrix_set(xs, 1, 1, 0.5);
    gsl_matrix_set(xs, 2, 0, 0.03); gsl_matrix_set(xs, 2, 1, 0.4);
    gsl_matrix_set(xs, 3, 0, 0.72); gsl_matrix_set(xs, 3, 1, 0.1);
    gsl_matrix_set(xs, 4, 0, 0.33); gsl_matrix_set(xs, 4, 1, 0.2);
    gsl_matrix_set(xs, 5, 0, 0.24); gsl_matrix_set(xs, 5, 1, 0.3);

    gsl_matrix* xis = gsl_matrix_alloc(3, 2);
    gsl_matrix_set(xis, 0, 0, -1.00); gsl_matrix_set(xis, 0, 1, -1.0);
    gsl_matrix_set(xis, 1, 0, +0.00); gsl_matrix_set(xis, 1, 1, +0.0);
    gsl_matrix_set(xis, 2, 0, +0.50); gsl_matrix_set(xis, 2, 1, +0.5);

    double globalBandwidth = 4.0;

    KernelType kernelType = STANDARD_GAUSSIAN;

    gsl_vector* actual = gsl_vector_alloc(numXs);
    gsl_vector* actualNumUsedPatterns = gsl_vector_alloc(numXs);

    gsl_vector* expected = gsl_vector_alloc(numXs);
    gsl_vector_set(expected, 0, 0.009694888542698);
    gsl_vector_set(expected, 1, 0.009536078921834);
    gsl_vector_set(expected, 2, 0.009608020510205);
    gsl_vector_set(expected, 3, 0.009466392462365);
    gsl_vector_set(expected, 4, 0.009603126595478);
    gsl_vector_set(expected, 5, 0.009602675165108);

    parzen(xs, xis, globalBandwidth, kernelType, 
        actual, actualNumUsedPatterns);

    CuAssertVectorEquals(tc, expected, actual, delta);

    /* Go back to the maximum number of threads */
    reset_omp();

    gsl_vector_free(actual);
    gsl_vector_free(expected);
    gsl_matrix_free(xs);
    gsl_matrix_free(xis);
}

CuSuite *ParzenGetSuite() {
    CuSuite *suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, testParzenSingleThreaded);
    SUITE_ADD_TEST(suite, testParzenMultiThreaded);
    return suite;
}