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
    printf("Array { data: %p, numberOfColumns: %2d, columnLength: %4d, row stride: %4d, col stride: %4d}\n",
    array->data, array->dimensionality, array->length, array->rowStride, array->colStride);
    double* currentElement = array->data;

    for (int i = 0;
         i < array->length;
         ++i, currentElement += array->rowStride) {
        arrayPrintElement(currentElement, array->dimensionality);
    }
    printf("\n");
}

void arrayPrintElement(double *element, int dimension){
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

double* arrayGetRowView(Array *array, int rowIdx){
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

ArrayColumns getColumns(Array *array) {
    ArrayColumns columns = arrayColumnsAllocate(array->dimensionality, array->length);
    for (int columnElementIdx = 0, dataIdx = 0; columnElementIdx < columns.columnLength; ++columnElementIdx) {
        for (int  columnIdx = 0; columnIdx < columns.numberOfColumns; ++columnIdx, dataIdx++) {
            columns.data[columnIdx][columnElementIdx] = array->data[dataIdx];
        }
    }
    return columns;
}

void arrayColumnsFree(ArrayColumns *matrix) {
    for(int i = 0; i < matrix->numberOfColumns; i++){
        free(matrix->data[i]);
    }
    free(matrix->data);
}

ArrayColumns arrayColumnsAllocate(int numberOfColumns, int columnLength) {
    double** data = malloc(numberOfColumns * sizeof(double*));
    for(int i = 0; i < numberOfColumns; i++){
        data[i] = malloc(columnLength * sizeof(double));
    }
    ArrayColumns matrix= {
            .data = data,
            .numberOfColumns = numberOfColumns,
            .columnLength = columnLength
    };
    return matrix;
}


void arrayColumnsPrint(ArrayColumns *matrix) {
    printf("ArrayColumns{ data: %p, numberOfColumns: %2d, columnLength: %4d}\n",
           matrix->data, matrix->numberOfColumns, matrix->columnLength);
    for (int i = 0; i < matrix->numberOfColumns; ++i) {
        arrayColumnsPrintColumn(matrix->data[i], matrix->columnLength);
    }
    printf("\n");
}



gsl_vector_view arrayGetGSLVectorView(Array *array) {
    size_t vector_length = (size_t) ((array->dimensionality > array->length) ? array->dimensionality : array->length);
    gsl_vector_view result = gsl_vector_view_array(array->data, vector_length);
    return result;
}

gsl_matrix_view arrayGetGSLMatrixView(Array* array){
    size_t num_rows = (size_t) array->length;
    size_t num_cols = (size_t) array->dimensionality;

    return gsl_matrix_view_array(array->data, num_rows, num_cols);
}

gsl_matrix *arrayCopyToGSLMatrix(Array *array) {
    gsl_matrix_view arrayView = arrayGetGSLMatrixView(array);
    return gsl_matrix_view_copy_to_gsl_matrix(arrayView);
}

void arrayColumnsPrintColumn(double *row, int length) {
    printf("[ ");
    for (int i = 0; i < length; ++i) {
        printf("%f ", row[i]);
    }
    printf("]\n");
}
