#include "covariancematrix.ih"


void computeCovarianceMatrix(Array *patterns, Array *covarianceMatrix) {
//http://stackoverflow.com/a/3307381/1357229
    double covariance;

    ColumnFist2DArray matrix = toColumnWiseMatrix(covarianceMatrix);

    columnFirst2DArrayPrint(&matrix);

    for(int colA = 0; colA < covarianceMatrix->dimensionality; colA++){
        arraySetElement(
                covarianceMatrix, colA, colA,
                computeVariance(matrix.data[colA], patterns->length)
        );

        for(int colB = colA  + 1; colB < covarianceMatrix->dimensionality; colB++){
            covariance = computeCovariance(matrix.data[colA], matrix.data[colB], patterns->length);
            arraySetElement(covarianceMatrix, colA, colB, covariance);
            arraySetElement(covarianceMatrix, colB, colA, covariance);
        }
    }
    columnFist2DArrayFree(&matrix);
}

double computeCovariance(double *columnA, double *columnB, int length) {
    double* someVector = elementWiseMultiplication(columnA, columnB, length);
    double termA = computeMean(someVector, length);
    double termB = computeMean(columnA, length) * computeMean(columnB, length);
    return termA - termB;
}

double * elementWiseMultiplication(double *vectorA, double *vectorB, int length) {
    double* product = malloc(length * sizeof(double));
    for(int i = 0; i < length; i++){
        product[i] = vectorA[i] * vectorB[i];
    }
    return product;
}

double computeVariance(double *data, int length) {
    double variance = 0;
    double mean = computeMean(data, length);
    for(int i = 0; i<length; i++){
        variance += ((data[i]) - mean) * ((data[i]) - mean);
    }
    variance = variance / length;
    return variance;
}

double computeMean(double *data, int length) {
    double mean = 0;
    for(int i = 0; i < length; i++){
        mean += data[i];
    }
    return mean / length;
}
