#include "covariancematrix.ih"


void computeCovarianceMatrix(Array *patterns, Array *covarianceMatrix) {
//http://stackoverflow.com/a/3307381/1357229
    double covariance;
    double* means;

    ArrayColumns columns = getColumns(patterns);

    means = computeMeans(&columns);

    for(int colA = 0; colA < columns.numberOfColumns; colA++){
        arraySetElement(
                covarianceMatrix, colA, colA,
                computeVariance(columns.data[colA], means[colA], patterns->length)
        );

        for(int colB = colA  + 1; colB < columns.numberOfColumns; colB++){
            covariance = computeCovariance(
                    columns.data[colA], means[colA],
                    columns.data[colB], means[colB],
                    columns.columnLength);
            arraySetElement(covarianceMatrix, colA, colB, covariance);
            arraySetElement(covarianceMatrix, colB, colA, covariance);
        }
    }
    arrayColumnsFree(&columns);
    free(means);
}

double* computeMeans(ArrayColumns* matrix){
    double* means = malloc(matrix->numberOfColumns * sizeof(double));
    for (int i = 0; i < matrix->numberOfColumns; ++i) {
        means[i] = computeMean(matrix->data[i], matrix->columnLength);
    }
    return means;
}

double computeCovariance(double *columnA, double meanA, double *columnB, double meanB, int length) {
    double* someVector = elementWiseMultiplication(columnA, columnB, length);
    double covariance = computeMean(someVector, length) - (meanA * meanB);
    free(someVector);
    return covariance;
}

double * elementWiseMultiplication(double *vectorA, double *vectorB, int length) {
    double* product = malloc(length * sizeof(double));
    for(int i = 0; i < length; i++){
        product[i] = vectorA[i] * vectorB[i];
    }
    return product;
}

double computeVariance(double *data, double mean, int length) {
    double variance = 0;
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
