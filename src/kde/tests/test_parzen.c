#include "test_parzen.h"

#include "../parzen.h"

#include <stdio.h>

#include "../../lib/CuTestUtils.h"
#include "../../test_constants.h"

#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_matrix.h>
#include "../kernels/kernels.h"


void testParzen(CuTest *tc){
	gsl_matrix* xs = gsl_matrix_alloc(2, 2);
    gsl_matrix_set(xs, 0, 0, 0.00); gsl_matrix_set(xs, 0, 1, 0.0);
    gsl_matrix_set(xs, 1, 0, 0.25); gsl_matrix_set(xs, 1, 1, 0.5);

	gsl_matrix* xis = gsl_matrix_alloc(3, 2);
    gsl_matrix_set(xis, 0, 0, -1.00); gsl_matrix_set(xis, 0, 1, -1.0);
    gsl_matrix_set(xis, 1, 0, +0.00); gsl_matrix_set(xis, 1, 1, +0.0);
    gsl_matrix_set(xis, 2, 0, +0.50); gsl_matrix_set(xis, 2, 1, +0.5);

    double globalBandwidth = 4.0;

    SymmetricKernel kernel = selectSymmetricKernel(STANDARD_GAUSSIAN);

    gsl_vector* actual = gsl_vector_alloc(2);

    gsl_vector* expected = gsl_vector_alloc(2);
    gsl_vector_set(expected, 0, 0.0096947375);
    gsl_vector_set(expected, 1, 0.0095360625);

    parzen(xs, xis, globalBandwidth, kernel, actual);

    CuAssertVectorEquals(tc, expected, actual, delta);

    gsl_vector_free(actual);
    gsl_vector_free(expected);
    gsl_matrix_free(xs);
    gsl_matrix_free(xis);
}

CuSuite *ParzenGetSuite() {
    CuSuite *suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, testParzen);
    return suite;
}