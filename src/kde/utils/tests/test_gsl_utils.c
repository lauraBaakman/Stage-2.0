#include "test_gsl_utils.h"

#include "../../../lib/CuTestUtils.h"
#include "../../../test_utils.h"

#include "../gsl_utils.h"

void testComputeRowMeans(CuTest *tc){
    gsl_matrix* matrix = gsl_matrix_alloc(3, 2);
    gsl_matrix_set(matrix, 0, 0, 5); gsl_matrix_set(matrix, 0, 1, 4);
    gsl_matrix_set(matrix, 1, 0, 3); gsl_matrix_set(matrix, 1, 1, 2);
    gsl_matrix_set(matrix, 2, 0, 7); gsl_matrix_set(matrix, 2, 1, 9);

    gsl_vector* expected = gsl_vector_alloc(3);
    gsl_vector_set(expected, 0, 4.5);
    gsl_vector_set(expected, 1, 2.5);
    gsl_vector_set(expected, 2, 8.0);

    gsl_vector* actual = gsl_vector_alloc(3);
    gsl_matrix_compute_row_means(matrix, actual);

    CuAssertVectorEquals(tc, expected, actual, delta);

    gsl_matrix_free(matrix);
    gsl_vector_free(actual);
    gsl_vector_free(expected);
}

void testComputeColMeans(CuTest *tc){
    gsl_matrix* matrix = gsl_matrix_alloc(3, 2);
    gsl_matrix_set(matrix, 0, 0, 5); gsl_matrix_set(matrix, 0, 1, 4);
    gsl_matrix_set(matrix, 1, 0, 3); gsl_matrix_set(matrix, 1, 1, 2);
    gsl_matrix_set(matrix, 2, 0, 7); gsl_matrix_set(matrix, 2, 1, 9);

    gsl_vector* expected = gsl_vector_alloc(2);
    gsl_vector_set(expected, 0, 5.0);
    gsl_vector_set(expected, 1, 5.0);

    gsl_vector* actual = gsl_vector_alloc(2);
    gsl_matrix_compute_col_means(matrix, actual);

    CuAssertVectorEquals(tc, expected, actual, delta);

    gsl_matrix_free(matrix);
    gsl_vector_free(actual);
    gsl_vector_free(expected);
}

CuSuite *GSLUtilsGetSuite() {
    CuSuite *suite = CuSuiteNew();
	SUITE_ADD_TEST(suite, testComputeRowMeans);
    SUITE_ADD_TEST(suite, testComputeColMeans);
    return suite;
}
