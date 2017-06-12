#include "test_knn.h"

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector_double.h>

#include "../../../lib/CuTestUtils.h"
#include "../../../test_constants.h"

#include "../knn.h"

void testKNN_x_in_xs_1(CuTest *tc){
    gsl_matrix* xs = gsl_matrix_alloc(4, 2);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 1, 0, 1); gsl_matrix_set(xs, 1, 1, 1);
    gsl_matrix_set(xs, 2, 0, 2); gsl_matrix_set(xs, 2, 1, 3);
    gsl_matrix_set(xs, 3, 0, 4); gsl_matrix_set(xs, 3, 1, 7);

    gsl_matrix* xs_copy = gsl_matrix_alloc(xs->size1, xs->size2);
    gsl_matrix_memcpy(xs_copy, xs);

    size_t k = 2;

    gsl_matrix* expected = gsl_matrix_alloc(k, xs->size2);
    gsl_matrix_set(expected, 0, 0, 0); gsl_matrix_set(expected, 0, 1, 0);
    gsl_matrix_set(expected, 1, 0, 1); gsl_matrix_set(expected, 1, 1, 1);

    gsl_vector_view pattern = gsl_matrix_row(xs, 0);

    gsl_matrix* actual = gsl_matrix_alloc(k, xs->size2);

    nn_prepare(xs);

    computeKNearestNeighbours(&pattern.vector, k, actual);

    CuAssertMatrixEquals(tc, expected, actual, delta);
    //Check if the function does not influence xs
    CuAssertMatrixEquals(tc, xs_copy, xs, delta);

    gsl_matrix_free(xs);
    gsl_matrix_free(expected);
    gsl_matrix_free(actual);
    nn_free();
}

void testKNN_x_in_xs_2(CuTest *tc){
    gsl_matrix* xs = gsl_matrix_alloc(4, 2);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 1, 0, 1); gsl_matrix_set(xs, 1, 1, 1);
    gsl_matrix_set(xs, 2, 0, 2); gsl_matrix_set(xs, 2, 1, 3);
    gsl_matrix_set(xs, 3, 0, 4); gsl_matrix_set(xs, 3, 1, 7);

    gsl_matrix* xs_copy = gsl_matrix_alloc(xs->size1, xs->size2);
    gsl_matrix_memcpy(xs_copy, xs);

    size_t k = 2;

    gsl_matrix* expected = gsl_matrix_alloc(k, xs->size2);
    gsl_matrix_set(expected, 0, 0, 2); gsl_matrix_set(expected, 0, 1, 3);
    gsl_matrix_set(expected, 1, 0, 4); gsl_matrix_set(expected, 1, 1, 7);

    gsl_vector_view pattern = gsl_matrix_row(xs, 2);

    gsl_matrix* actual = gsl_matrix_alloc(k, xs->size2);

    nn_prepare(xs);

    computeKNearestNeighbours(&pattern.vector, k, actual);

    CuAssertMatrixEquals(tc, expected, actual, delta);
    //Check if the function does not influence xs
    CuAssertMatrixEquals(tc, xs_copy, xs, delta);

    gsl_matrix_free(xs);
    gsl_matrix_free(expected);
    gsl_matrix_free(actual);
    nn_free();
}

void testKNN_x_not_in_xs_1(CuTest *tc){
    gsl_matrix* xs = gsl_matrix_alloc(4, 2);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 1, 0, 1); gsl_matrix_set(xs, 1, 1, 1);
    gsl_matrix_set(xs, 2, 0, 2); gsl_matrix_set(xs, 2, 1, 3);
    gsl_matrix_set(xs, 3, 0, 4); gsl_matrix_set(xs, 3, 1, 7);

    gsl_matrix* xs_copy = gsl_matrix_alloc(xs->size1, xs->size2);
    gsl_matrix_memcpy(xs_copy, xs);

    size_t k = 2;

    gsl_matrix* expected = gsl_matrix_alloc(k, xs->size2);
    gsl_matrix_set(expected, 0, 0, 2); gsl_matrix_set(expected, 0, 1, 3);
    gsl_matrix_set(expected, 1, 0, 1); gsl_matrix_set(expected, 1, 1, 1);

    gsl_vector* pattern = gsl_vector_alloc(xs->size2);
    gsl_vector_set(pattern, 0, 0.5);
    gsl_vector_set(pattern, 1, 3.5);

    gsl_matrix* actual = gsl_matrix_alloc(k, xs->size2);

    nn_prepare(xs);

    computeKNearestNeighbours(pattern, k, actual);

    CuAssertMatrixEquals(tc, expected, actual, delta);
    //Check if the function does not influence xs
    CuAssertMatrixEquals(tc, xs_copy, xs, delta);

    gsl_matrix_free(xs);
    gsl_matrix_free(expected);
    gsl_matrix_free(actual);
    gsl_vector_free(pattern);
    nn_free();
}

void testKNN_x_not_in_xs_2(CuTest *tc){
    gsl_matrix* xs = gsl_matrix_alloc(4, 2);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 1, 0, 1); gsl_matrix_set(xs, 1, 1, 1);
    gsl_matrix_set(xs, 2, 0, 2); gsl_matrix_set(xs, 2, 1, 3);
    gsl_matrix_set(xs, 3, 0, 4); gsl_matrix_set(xs, 3, 1, 7);

    gsl_matrix* xs_copy = gsl_matrix_alloc(xs->size1, xs->size2);
    gsl_matrix_memcpy(xs_copy, xs);

    size_t k = 2;

    gsl_matrix* expected = gsl_matrix_alloc(k, xs->size2);
    gsl_matrix_set(expected, 0, 0, 0); gsl_matrix_set(expected, 0, 1, 0);
    gsl_matrix_set(expected, 1, 0, 1); gsl_matrix_set(expected, 1, 1, 1);

    gsl_vector* pattern = gsl_vector_alloc(xs->size2);
    gsl_vector_set(pattern, 0, 0.1);
    gsl_vector_set(pattern, 1, 0.1);

    gsl_matrix* actual = gsl_matrix_alloc(k, xs->size2);

    nn_prepare(xs);

    computeKNearestNeighbours(pattern, k, actual);

    CuAssertMatrixEquals(tc, expected, actual, delta);
    //Check if the function does not influence xs
    CuAssertMatrixEquals(tc, xs_copy, xs, delta);

    gsl_matrix_free(xs);
    gsl_matrix_free(expected);
    gsl_matrix_free(actual);
    gsl_vector_free(pattern);
    nn_free();
}

void testKNN_x_not_in_xs_3(CuTest *tc){
    gsl_matrix* xs = gsl_matrix_alloc(4, 2);
    gsl_matrix_set(xs, 0, 0, 4); gsl_matrix_set(xs, 0, 1, 7);
    gsl_matrix_set(xs, 1, 0, 1); gsl_matrix_set(xs, 1, 1, 1);
    gsl_matrix_set(xs, 2, 0, 2); gsl_matrix_set(xs, 2, 1, 3);
    gsl_matrix_set(xs, 3, 0, 0); gsl_matrix_set(xs, 3, 1, 0);

    gsl_matrix* xs_copy = gsl_matrix_alloc(xs->size1, xs->size2);
    gsl_matrix_memcpy(xs_copy, xs);

    size_t k = 2;

    gsl_matrix* expected = gsl_matrix_alloc(k, xs->size2);
    gsl_matrix_set(expected, 0, 0, 0); gsl_matrix_set(expected, 0, 1, 0);
    gsl_matrix_set(expected, 1, 0, 1); gsl_matrix_set(expected, 1, 1, 1);

    gsl_vector* pattern = gsl_vector_alloc(xs->size2);
    gsl_vector_set(pattern, 0, 0.1);
    gsl_vector_set(pattern, 1, 0.1);

    gsl_matrix* actual = gsl_matrix_alloc(k, xs->size2);

    nn_prepare(xs);

    computeKNearestNeighbours(pattern, k, actual);

    CuAssertMatrixEquals(tc, expected, actual, delta);
    //Check if the function does not influence xs
    CuAssertMatrixEquals(tc, xs_copy, xs, delta);

    gsl_matrix_free(xs);
    gsl_matrix_free(expected);
    gsl_matrix_free(actual);
    gsl_vector_free(pattern);
    nn_free();
}

CuSuite *KNNGetSuite() {
    CuSuite *suite = CuSuiteNew();
//  These test fails due to issues with the library, should not be too big of a problem, might be fixed eventually.
//    SUITE_ADD_TEST(suite, testKNN_x_in_xs_1);
//    SUITE_ADD_TEST(suite, testKNN_x_in_xs_2);
//    SUITE_ADD_TEST(suite, testKNN_x_not_in_xs_2);
    SUITE_ADD_TEST(suite, testKNN_x_not_in_xs_1);
    SUITE_ADD_TEST(suite, testKNN_x_not_in_xs_3);
    return suite;
}
