#include <stdio.h>
#include "gsl_utils.h"

int gsl_matrix_print(FILE *f, const gsl_matrix *m) {
    int status, n = 0;

    for (size_t i = 0; i < m->size1; i++) {
        for (size_t j = 0; j < m->size2; j++) {
            if ((status = fprintf(f, "%# .3g ", gsl_matrix_get(m, i, j))) < 0)
                return -1;
            n += status;
        }

        if ((status = fprintf(f, "\n")) < 0)
            return -1;
        n += status;
    }

    return n;
}

int gsl_vector_print(FILE *f, const gsl_vector *vector) {
    int status, n = 0;

    for (size_t i = 0; i < vector->size; i++) {
        if ((status = fprintf(f, "%# .3g ", gsl_vector_get(vector, i))) < 0) return -1;
        n += status;
    }
    if ((status = fprintf(f, "\n")) < 0)
        return -1;
    return n;
}

int gsl_permutation_print(FILE *f, const gsl_permutation *permutation){
    int status, n= 0;
    for (size_t i = 0; i < permutation->size; i++) {
        if ((status = fprintf(f, "%.3lu ", gsl_permutation_get(permutation, i))) < 0) return -1;
        n += status;
    }
    if ((status = fprintf(f, "\n")) < 0)
        return -1;
    return n;
}

gsl_matrix* gsl_matrix_view_copy_to_gsl_matrix(gsl_matrix_view origin) {
    gsl_matrix *copy = gsl_matrix_alloc(origin.matrix.size1, origin.matrix.size2);
    gsl_matrix_memcpy(copy, &origin.matrix);
    return copy;
}

gsl_vector *gsl_subtract(gsl_vector *termA, gsl_vector *termB, gsl_vector *result) {
    gsl_vector_memcpy(result, termA);
    gsl_vector_sub(result, termB);

    return result;
}

void gsl_matrix_compute_row_means(gsl_matrix* data, gsl_vector* means){
    gsl_vector_view row;
    double mean;

    for(size_t i = 0; i < data->size1; i++){
        row = gsl_matrix_row(data, i);
        mean = gsl_stats_mean(row.vector.data, row.vector.stride, row.vector.size);
        gsl_vector_set(means, i, mean);
    }
}

void gsl_matrix_compute_col_means(gsl_matrix* data, gsl_vector* means){
    gsl_vector_view col;
    double mean;

    for(size_t i = 0; i < data->size2; i++){
        col = gsl_matrix_column(data, i);
        mean = gsl_stats_mean(col.vector.data, col.vector.stride, col.vector.size);
        gsl_vector_set(means, i, mean);
    }
}

gsl_matrix** gsl_matrices_alloc(size_t size1, size_t size2, int numMatrices){
    gsl_matrix** matrices = (gsl_matrix**) malloc(numMatrices * sizeof(gsl_matrix*));
    for(int i = 0; i < numMatrices; i++)   {
        matrices[i] = gsl_matrix_alloc(size1, size2);
    }
    return matrices;
}

void gsl_matrices_free(gsl_matrix** matrices, int numMatrices){
    for(int i = 0; i < numMatrices; i++)   {
        gsl_matrix_free(matrices[i]);
    }
    free(matrices);
}

gsl_vector** gsl_vectors_alloc(size_t size1, int numVectors){
    gsl_vector** vectors = (gsl_vector**) malloc(numVectors * sizeof(gsl_vector*));
    for(int i = 0; i < numVectors; i++)   {
        vectors[i] = gsl_vector_alloc(size1);
    }
    return vectors;
}

void gsl_vectors_free(gsl_vector** vectors, int numVectors){
    for(int i = 0; i < numVectors; i++)   {
        gsl_vector_free(vectors[i]);
    }
    free(vectors);
}

gsl_permutation** gsl_permutations_alloc(size_t size, int numPermutations){
    gsl_permutation** permutations = (gsl_permutation**) malloc(numPermutations * sizeof(gsl_permutation*));
    for(int i = 0; i < numPermutations; i++)   {
        permutations[i] = gsl_permutation_alloc(size);
    }
    return permutations;
}

void gsl_permutations_free(gsl_permutation** permutations, int numPermutations){
    for(int i = 0; i < numPermutations; i++)   {
        gsl_permutation_free(permutations[i]);
    }
    free(permutations);
}