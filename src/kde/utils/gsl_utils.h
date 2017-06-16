#ifndef KERNELS_GSL_UTILS_H
#define KERNELS_GSL_UTILS_H

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_permutation.h>

gsl_matrix* gsl_matrix_view_copy_to_gsl_matrix(gsl_matrix_view origin);

int gsl_matrix_print(FILE *f, const gsl_matrix *m);

int gsl_vector_print(FILE *f, const gsl_vector *vector);

int gsl_permutation_print(FILE *f, const gsl_permutation *permutation);

gsl_vector* gsl_subtract(gsl_vector *termA, gsl_vector *termB, gsl_vector *result);

gsl_matrix** gsl_matrices_alloc(size_t size1, size_t size2, int numMatrices);
void gsl_matrices_free(gsl_matrix** matrices, int numMatrices);

gsl_vector** gsl_vectors_alloc(size_t size, int numVectors);
void gsl_vectors_free(gsl_vector** vectors, int numVectors);

gsl_permutation** gsl_permutations_alloc(size_t size, int numPermutations);
void gsl_permutations_free(gsl_permutation** permutations, int numPermutations);

#endif //KERNELS_GSL_UTILS_H
