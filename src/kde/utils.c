//
// Created by Laura Baakman on 20/01/2017.
//
#include "utils.ih"

Array arrayBuildFromPyArray(PyArrayObject *arrayObject){

    double* data = (double *)PyArray_DATA(arrayObject);

    int dimensionality = determine_dimensionality(arrayObject);
    int length = (int)PyArray_DIM(arrayObject, 0);

    int rowStride = (int)PyArray_STRIDE (arrayObject, 0) / (int)PyArray_ITEMSIZE(arrayObject);
    int colStride = (int)PyArray_STRIDE (arrayObject, 1) / (int)PyArray_ITEMSIZE(arrayObject);

    Array array = {
            .data = data,
            .dimensionality = dimensionality,
            .length = length,
            .rowStride= rowStride,
            .colStride = colStride
    };
    return array;
}

int determine_dimensionality(PyArrayObject* arrayObject){
    int num_dimensions = PyArray_NDIM(arrayObject);
    if (num_dimensions == 1) {
        return 1;
    } else {
        return (int)PyArray_DIM(arrayObject, 1);
    }
}

void arrayPrint(Array *array){
    printf("Array { data: %p, dimensionality: %2d, length: %4d, row stride: %4d, col stride: %4d}\n",
    array->data, array->dimensionality, array->length, array->rowStride, array->colStride);
    double* currentElement = array->data;

    for (int i = 0;
         i < array->length;
         ++i, currentElement += array->rowStride) {
        printElement(currentElement, array->dimensionality);
    }
    printf("\n");
}

void printElement(double* element, int dimension){
    printf("[ ");
    for (int i = 0; i < dimension; ++i) {
        printf("%f ", element[i]);
    }
    printf("]\n");
}

void arraySetDiagonalToZero(Array *array){
    double* currentRow = array->data;
    for (int i = 0;
         i < array->length;
         ++i, currentRow+= array->rowStride) {
        currentRow[i] = 0;
    }
}

void arraySetElement(Array* array, int rowIdx, int colIdx, double value){
    double* row = array->data + rowIdx * array->rowStride;
    row[colIdx] = value;
}

double* arrayGetRow(Array* array, int rowIdx){
    double* row = array->data + rowIdx * array->rowStride;
    return row;
}

void arraySetRow(Array* array, int rowIdx, double* values){
    for(int i = 0; i < array->dimensionality; i++){
        arraySetElement(array, rowIdx, i, values[i]);
    }
}


double* scalePattern(double* pattern, double* dataPoint, double* scaledPattern, int dimensionality, double windowWidth){
    for (int i = 0; i < dimensionality; ++i) {
        scaledPattern[i] = (pattern[i] - dataPoint[i]) / windowWidth;
    }
    return scaledPattern;
}