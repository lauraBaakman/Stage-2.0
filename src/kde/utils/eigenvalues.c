#include "eigenvalues.h"

void computeEigenValues(gsl_matrix *matrix, gsl_vector *eigenValues, gsl_matrix* eigenVectors) {
    size_t matrixOrder = matrix->size1;

    gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc(matrixOrder);

    gsl_eigen_symmv(matrix, eigenValues, eigenVectors, w);

    gsl_eigen_symmv_free(w);
}
