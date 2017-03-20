#include <gsl/gsl_eigen.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector_double.h>

#include "eigenvalues.h"
#include "../../../../../../../usr/local/include/gsl/gsl_matrix.h"

void computeEigenValues(Array *data, Array *eigenValues) {
    size_t matrixOrder = (size_t) data->dimensionality;

    gsl_matrix_view matrixView = gsl_matrix_view_array (data->data, matrixOrder, matrixOrder);
    gsl_vector_view eigenValuesView = gsl_vector_view_array(eigenValues->data, (size_t) eigenValues->length);

    gsl_eigen_symm_workspace * w = gsl_eigen_symm_alloc(matrixOrder);

    gsl_eigen_symm(&matrixView.matrix, &eigenValuesView.vector, w);

    gsl_eigen_symm_free (w);
}

gsl_vector *computeEigenValues2(gsl_matrix *matrix) {
    size_t matrixOrder = matrixOrder = matrix->size1;

    gsl_vector* eigenValues = gsl_vector_alloc(matrixOrder);
    gsl_eigen_symm_workspace * w = gsl_eigen_symm_alloc(matrixOrder);

    gsl_eigen_symm(matrix, eigenValues, w);

    gsl_eigen_symm_free (w);

    return eigenValues;
}
