#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector_double.h>
#include "eigenvalues.h"

void computeEigenValues(Array *data, Array *eigenValues) {
    size_t matrixOrder = (size_t) data->dimensionality;

    gsl_matrix_view gslMatrix = gsl_matrix_view_array (data->data, matrixOrder, matrixOrder);
    gsl_vector_view gslEigenValues = gsl_vector_view_array(eigenValues->data, matrixOrder);
    gsl_eigen_symm_workspace * workspace = gsl_eigen_symm_alloc (matrixOrder);

    gsl_eigen_symm(&gslMatrix.matrix, &gslEigenValues.vector, workspace);

    gsl_eigen_symm_free (workspace);
}
