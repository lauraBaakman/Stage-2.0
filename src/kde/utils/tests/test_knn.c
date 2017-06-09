#include "test_knn.h"

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector_double.h>

#include "../../../lib/CuTestUtils.h"
#include "../../../test_constants.h"

#include "../knn.h"

void testKNNOld(CuTest *tc){
	gsl_matrix* xs = gsl_matrix_alloc(4, 2);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 1, 0, 1); gsl_matrix_set(xs, 1, 1, 1);
    gsl_matrix_set(xs, 2, 0, 2); gsl_matrix_set(xs, 2, 1, 3);
    gsl_matrix_set(xs, 3, 0, 4); gsl_matrix_set(xs, 3, 1, 7);

    int k = 2;

    gsl_matrix* expected = gsl_matrix_alloc(2, 2);
    gsl_matrix_set(expected, 0, 0, 0); gsl_matrix_set(expected, 0, 1, 0);
    gsl_matrix_set(expected, 1, 0, 1); gsl_matrix_set(expected, 1, 1, 1);

    gsl_matrix* actual = gsl_matrix_alloc(2, 2);

    size_t patternIdx = 0;

    nn_prepare(xs);

    computeKNearestNeighboursOld(k, patternIdx, xs, actual);

    CuAssertMatrixEquals(tc, expected, actual, delta);

    gsl_matrix_free(xs);
    gsl_matrix_free(expected);
    gsl_matrix_free(actual);
    nn_free();
}

void testKNN_x_in_xs(CuTest *tc){
    gsl_matrix* xs = gsl_matrix_alloc(4, 2);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 1, 0, 1); gsl_matrix_set(xs, 1, 1, 1);
    gsl_matrix_set(xs, 2, 0, 2); gsl_matrix_set(xs, 2, 1, 3);
    gsl_matrix_set(xs, 3, 0, 4); gsl_matrix_set(xs, 3, 1, 7);

    size_t k = 2;

    gsl_matrix* expected = gsl_matrix_alloc(k, xs->size2);
    gsl_matrix_set(expected, 0, 0, 0); gsl_matrix_set(expected, 0, 1, 0);
    gsl_matrix_set(expected, 1, 0, 1); gsl_matrix_set(expected, 1, 1, 1);

    gsl_vector_view pattern = gsl_matrix_row(xs, 0);

    gsl_matrix* actual = gsl_matrix_alloc(k, xs->size2);

    nn_prepare(xs);

    computeKNearestNeighbours(&pattern.vector, k, actual);

    CuAssertMatrixEquals(tc, expected, actual, delta);

    gsl_matrix_free(xs);
    gsl_matrix_free(expected);
    gsl_matrix_free(actual);
    nn_free();
}

void testKNN_x_not_in_xs(CuTest *tc){
    gsl_matrix* xs = gsl_matrix_alloc(4, 2);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 1, 0, 1); gsl_matrix_set(xs, 1, 1, 1);
    gsl_matrix_set(xs, 2, 0, 2); gsl_matrix_set(xs, 2, 1, 3);
    gsl_matrix_set(xs, 3, 0, 4); gsl_matrix_set(xs, 3, 1, 7);

    size_t k = 2;

    gsl_matrix* expected = gsl_matrix_alloc(k, xs->size2);
    gsl_matrix_set(expected, 0, 0, 1); gsl_matrix_set(expected, 0, 1, 1);
    gsl_matrix_set(expected, 1, 0, 2); gsl_matrix_set(expected, 1, 1, 3);

    gsl_vector* pattern = gsl_vector_alloc(xs->size2);
    gsl_vector_set(pattern, 0, 0.5);
    gsl_vector_set(pattern, 0, 3.5);

    gsl_matrix* actual = gsl_matrix_alloc(k, xs->size2);

    nn_prepare(xs);

    computeKNearestNeighbours(pattern, k, actual);

    CuAssertMatrixEquals(tc, expected, actual, delta);

    gsl_matrix_free(xs);
    gsl_matrix_free(expected);
    gsl_matrix_free(actual);
    gsl_vector_free(pattern);
    nn_free();
}

CuSuite *KNNGetSuite() {
    CuSuite *suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, testKNNOld);
    SUITE_ADD_TEST(suite, testKNN_x_in_xs);
    SUITE_ADD_TEST(suite, testKNN_x_not_in_xs);
    return suite;
}
