#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
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
        if ((status = fprintf(f, "%# .3g ", gsl_permutation_get(permutation, i))) < 0) return -1;
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