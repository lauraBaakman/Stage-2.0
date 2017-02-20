#ifndef UTILS_H
#define UTILS_H

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

typedef struct Array{
    double* data;
    int length;
    int dimensionality;
    int rowStride;
    int colStride;
} Array;

Array arrayBuildFromPyArray(PyArrayObject *arrayObject);
void arrayPrint(Array *array);
void arraySetDiagonalToZero(Array *array);
void arraySetElement(Array* array, int rowIdx, int colIdx, double value);
void arraySetRow(Array* array, int rowIdx, double* values);
double* arrayGetRow(Array* array, int rowIdx);


double* scalePattern(double* pattern, double* dataPoint, double* scaledPattern, int dimensionality, double windowWidth);

#endif //UTILS_H
