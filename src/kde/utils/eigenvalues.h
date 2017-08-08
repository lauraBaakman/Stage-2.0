#ifndef KERNELS_EIGENVALUES_H
#define KERNELS_EIGENVALUES_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>

void computeEigenValues(gsl_matrix *matrix, gsl_vector *eigenValues, gsl_matrix* eigenVectors);


#endif //KERNELS_EIGENVALUES_H
