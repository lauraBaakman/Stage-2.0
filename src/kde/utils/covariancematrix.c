#include <gsl/gsl_matrix.h>
#include <gsl/gsl_statistics_double.h>
#include "covariancematrix.ih"

void computeCovarianceMatrix(gsl_matrix *patterns, gsl_matrix *covarianceMatrix) {
    gsl_vector_view a, b;
    double variance;

    for(size_t i = 0; i < covarianceMatrix->size1; i++){
        for(size_t j = i; j < covarianceMatrix->size2; j++){

            a = gsl_matrix_column(patterns, i);
            b = gsl_matrix_column(patterns, j);

            variance = gsl_stats_covariance(a.vector.data, a.vector.stride,
                                            b.vector.data, b.vector.stride,
                                            b.vector.size);
            gsl_matrix_set(covarianceMatrix, i, j, variance);
            gsl_matrix_set(covarianceMatrix, j, i, variance);
        }
    }
}