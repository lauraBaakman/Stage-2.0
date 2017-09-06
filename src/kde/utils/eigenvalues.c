#include "eigenvalues.h"
#include "gsl_utils.h"

void computeEigenValues(gsl_matrix *matrix, gsl_vector *eigenValues, gsl_matrix* eigenVectors) {
    size_t matrixOrder = matrix->size1;

    gsl_matrix* changableInputMatrix = gsl_matrix_alloc(matrix->size1, matrix->size2);
    gsl_matrix_memcpy(changableInputMatrix, matrix);

    gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc(matrixOrder);

    gsl_eigen_symmv(changableInputMatrix, eigenValues, eigenVectors, w);

    gsl_matrix_free(changableInputMatrix);
    gsl_eigen_symmv_free(w);
}