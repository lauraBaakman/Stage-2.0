#ifndef UTILS_H
#define UTILS_H

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

typedef struct Array{
    double* data;
    int length;
    int dimensionality;
    int stride;
} Array;

Array arrayBuildFromPyArray(PyArrayObject *arrayObject);
void arrayPrint(Array *array);
void arraySetDiagonalToZero(Array *array);


double* scalePattern(double* pattern, double* dataPoint, double* scaledPattern, int dimensionality, double windowWidth);

#endif //UTILS_H
