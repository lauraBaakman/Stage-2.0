#include <gsl/gsl_eigen.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector_double.h>

#include "eigenvalues.h"

void computeEigenValues(Array *data, Array *eigenValues) {
    size_t matrixOrder = (size_t) data->dimensionality;

    gsl_matrix_view matrixView = gsl_matrix_view_array (data->data, matrixOrder, matrixOrder);
    
    gsl_vector_view eigenValuesView = gsl_vector_view_array(eigenValues->data, (size_t) eigenValues->length);
    gsl_matrix *evec = gsl_matrix_alloc (matrixOrder, matrixOrder);

    gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc (matrixOrder);

    gsl_eigen_symmv (&matrixView.matrix, &eigenValuesView.vector, evec, w);

    gsl_eigen_symmv_free (w);

    gsl_matrix_free (evec);
}
