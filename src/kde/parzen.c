//
// Created by Laura Baakman on 09/01/2017.
//

#include "parzen.h"

double parzen(double* pattern, int dimensionality, PyArrayObject* dataPoints, double windowWidth, double factor){
    int numDataPoints = (int)PyArray_DIM(dataPoints, 0);

    double* dataDataPoints = (double *)PyArray_DATA(dataPoints);

    int strideDataPoints = (int)PyArray_STRIDE (dataPoints, 0) / (int)PyArray_ITEMSIZE(dataPoints);

    double* currentDataPoint = dataDataPoints;

    double density = 0;
    double gaussianFactor = standardGaussianFactor(dimensionality);

    double* scaledPattern = (double *)malloc(sizeof(double) * dimensionality);

    for (int i = 0; i < numDataPoints; ++i, currentDataPoint += strideDataPoints) {
        scaledPattern = scalePattern(pattern, currentDataPoint, scaledPattern, dimensionality, windowWidth);
        density += standardGaussian(scaledPattern, dimensionality, gaussianFactor);
    }
    density *= factor;

    free(scaledPattern);
    return density;
}

double* scalePattern(double* pattern, double* dataPoint, double* scaledPattern, int dimensionality, double windowWidth){
    for (int i = 0; i < dimensionality; ++i) {
        scaledPattern[i] = (pattern[i] - dataPoint[i]) / windowWidth;
    }
    return scaledPattern;
}