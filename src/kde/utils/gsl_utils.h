#ifndef KERNELS_GSL_UTILS_H
#define KERNELS_GSL_UTILS_H

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_permutation.h>

gsl_matrix* gsl_matrix_view_copy_to_gsl_matrix(gsl_matrix_view origin);

int gsl_matrix_print(FILE *f, const gsl_matrix *m);

int gsl_vector_print(FILE *f, const gsl_vector *vector);

int gsl_permutation_print(FILE *f, const gsl_permutation *permutation);

#endif //KERNELS_GSL_UTILS_H
