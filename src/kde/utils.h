#ifndef UTILS_H
#define UTILS_H

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <gsl/gsl_matrix.h>

typedef struct Array{
    double* data;
    int length;
    int dimensionality;
    int rowStride;
    int colStride;
} Array;

typedef struct ArrayColumns{
    double** data;
    int columnLength;
    int numberOfColumns;
} ArrayColumns;

Array arrayBuildFromPyArray(PyArrayObject *arrayObject);
void arrayPrint(Array *array);
void arraySetDiagonalToZero(Array *array);
void arraySetElement(Array* array, int rowIdx, int colIdx, double value);
void arraySetRow(Array* array, int rowIdx, double* values);

gsl_vector_view arrayGetGSLVectorView(Array* array);

gsl_matrix_view arrayGetGSLMatrixView(Array* array);
gsl_matrix* arrayCopyToGSLMatrix(Array* array);

double* arrayGetRowView(Array *array, int rowIdx);

ArrayColumns getColumns(Array *array);
void arrayColumnsFree(ArrayColumns *matrix);
void arrayColumnsPrint(ArrayColumns *matrix);


gsl_matrix* gsl_matrix_view_copy_to_gsl_matrix(gsl_matrix_view origin);

int gsl_matrix_print(FILE *f, const gsl_matrix *m);

int gsl_vector_print(FILE *f, const gsl_vector *vector);

double* scalePattern(double* pattern, double* dataPoint, double* scaledPattern, int dimensionality, double windowWidth);

#endif //UTILS_H
