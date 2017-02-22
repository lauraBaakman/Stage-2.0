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

typedef struct ColumnFist2DArray{
    double** data;
    int length;
    int dimensionality;
} ColumnFist2DArray;

Array arrayBuildFromPyArray(PyArrayObject *arrayObject);
void arrayPrint(Array *array);
void arraySetDiagonalToZero(Array *array);
void arraySetElement(Array* array, int rowIdx, int colIdx, double value);
void arraySetRow(Array* array, int rowIdx, double* values);

double* arrayGetRowView(Array *array, int rowIdx);

ColumnFist2DArray toColumnWiseMatrix(Array *array);
void columnFist2DArrayFree(ColumnFist2DArray *matrix);
void columnFirst2DArrayPrint(ColumnFist2DArray *matrix);
void columnFirst2DArrayPrintColumn(double *row, int length);
ColumnFist2DArray columnFirst2DArrayAllocate(int length, int dimensionality);

double* scalePattern(double* pattern, double* dataPoint, double* scaledPattern, int dimensionality, double windowWidth);

#endif //UTILS_H
