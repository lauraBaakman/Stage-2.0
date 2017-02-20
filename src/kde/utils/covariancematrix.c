#include "covariancematrix.ih"


double computeMean(double **data, int length);

void computeCovarianceMatrix(Array *patterns, Array *covarianceMatrix) {
//http://stackoverflow.com/a/3307381/1357229
    double **columnA = (double**) malloc(patterns->length * sizeof(double*));
    double **columnB = (double**) malloc(patterns->length * sizeof(double*));

    double covariance;

    for(int row = 0; row < covarianceMatrix->dimensionality; row++){

        columnA = arrayGetColumn(patterns, row, columnA);
        arraySetElement(covarianceMatrix, row, row, computeVariance(columnA, patterns->length));

        for(int col = row  + 1; col < covarianceMatrix->dimensionality; col++){

            columnB = arrayGetColumn(patterns, col, columnB);

            covariance = computeCovariance(columnA, columnB, patterns->length);
            arraySetElement(covarianceMatrix, row, col, covariance);
            arraySetElement(covarianceMatrix, col, row, covariance);
        }
    }

    free(columnA);
    free(columnB);
}

double computeCovariance(double **columnA, double **columnB, int length) {
    return 1.0;
}

double computeVariance(double **data, int length) {
    double variance = 0;
    double mean = computeMean(data, length);
    for(int i = 0; i<length; i++){
        variance += (*(data[i]) - mean) * (*(data[i]) - mean);
    }
    variance = variance / length;
    return variance;
}

double computeMean(double **data, int length) {
    double mean = 0;
    for(int i = 0; i < length; i++){
        mean += *(data[i]);
    }
    return mean / length;
}
