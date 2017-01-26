//
// Created by Laura Baakman on 20/01/2017.
//

#include "utils.h"

Array buildArrayFromPyArray(PyArrayObject* arrayObject){

    double* data = (double *)PyArray_DATA(arrayObject);
    int length = (int)PyArray_DIM(arrayObject, 0);
    int dimensionality = (int)PyArray_DIM(arrayObject, 1);
    int stride = (int)PyArray_STRIDE (arrayObject, 0) / (int)PyArray_ITEMSIZE(arrayObject);

    Array array = {
            .data = data,
            .dimensionality = dimensionality,
            .length = length,
            .stride = stride
    };
    return array;
}

void printArray(Array* array){
    printf("Array { data: %p, dimensionality: %2d, length: %4d, stride: %4d}\n",
    array->data, array->dimensionality, array->length, array->stride);
}