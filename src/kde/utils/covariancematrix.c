#include "covariancematrix.ih"


void computeCovarianceMatrix(Array *patterns, Array *covarianceMatrix) {
//http://stackoverflow.com/a/3307381/1357229
    double* columnA = patterns->data;
    double* columnB = NULL;
    double covariance;
    for(int row = 0; row < covarianceMatrix->dimensionality; row++, columnA += patterns->colStride){
        arraySetElement(covarianceMatrix, row, row, computeVariance(columnA, patterns->length));
        columnB += patterns->colStride * (row + 1);
        for(int col = row  + 1; col < covarianceMatrix->dimensionality; col++, columnB+= patterns->colStride){
            covariance = computeCovariance(columnA, columnB, patterns->length);
            arraySetElement(covarianceMatrix, row, col, covariance);
            arraySetElement(covarianceMatrix, col, row, covariance);
        }
    }
}

double computeCovariance(double *columnA, double *columnB, int length) {
    return 1.0;
}

double computeVariance(double *pDouble, int length) {
    return 42.0;
}
