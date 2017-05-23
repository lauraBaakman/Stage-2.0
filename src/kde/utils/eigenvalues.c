#include "eigenvalues.h"

void computeEigenValues(gsl_matrix *matrix, gsl_vector *eigenValues) {
    size_t matrixOrder = matrix->size1;

    gsl_eigen_symm_workspace * w = gsl_eigen_symm_alloc(matrixOrder);

    gsl_eigen_symm(matrix, eigenValues, w);

    gsl_eigen_symm_free(w);
}
