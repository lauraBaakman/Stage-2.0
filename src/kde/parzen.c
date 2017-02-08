//
// Created by Laura Baakman on 09/01/2017.
//

#include "parzen.h"
#include "utils.h"

double parzen(double *pattern, Array *dataPoints, double windowWidth, double parzenFactor,
              KernelDensityFunction kernel, double kernelConstant){
    double* currentDataPoint = dataPoints->data;
    double density = 0;

    double* scaledPattern = (double *)malloc(sizeof(double) * dataPoints->dimensionality);

    for (int i = 0;
         i < dataPoints->length;
         ++i, currentDataPoint += dataPoints->stride)
    {
        scaledPattern = scalePattern(pattern, currentDataPoint, scaledPattern, dataPoints->dimensionality, windowWidth);
        density += kernel(scaledPattern, dataPoints->dimensionality, kernelConstant);
    }
    density *= parzenFactor;

    free(scaledPattern);
    return density;
}

double* scalePattern(double* pattern, double* dataPoint, double* scaledPattern, int dimensionality, double windowWidth){
    for (int i = 0; i < dimensionality; ++i) {
        scaledPattern[i] = (pattern[i] - dataPoint[i]) / windowWidth;
    }
    return scaledPattern;
}