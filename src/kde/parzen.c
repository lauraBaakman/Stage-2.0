//
// Created by Laura Baakman on 09/01/2017.
//

#include "parzen.h"
#include "utils.h"

double parzen_gaussian(double *pattern, Array *dataPoints, double windowWidth, double parzenFactor,
                       double gaussianFactor) {
    double* currentDataPoint = dataPoints->data;
    double density = 0;

    double* scaledPattern = (double *)malloc(sizeof(double) * dataPoints->dimensionality);

    for (int i = 0;
         i < dataPoints->length;
         ++i, currentDataPoint += dataPoints->stride)
    {
        scaledPattern = scalePattern(pattern, currentDataPoint, scaledPattern, dataPoints->dimensionality, windowWidth);
        density += standardGaussianPDF(scaledPattern, dataPoints->dimensionality, gaussianFactor);
    }
    density *= parzenFactor;

    free(scaledPattern);
    return density;
}

double parzen_epanechnikov(double *pattern, Array *dataPoints, double windowWidth, double parzenFactor,
                       double epanechnikovFactor) {
    double* currentDataPoint = dataPoints->data;
    double density = 0;

    double* scaledPattern = (double *)malloc(sizeof(double) * dataPoints->dimensionality);

    for (int i = 0;
         i < dataPoints->length;
         ++i, currentDataPoint += dataPoints->stride)
    {
        scaledPattern = scalePattern(pattern, currentDataPoint, scaledPattern, dataPoints->dimensionality, windowWidth);
        density += epanechnikovPDF(scaledPattern, dataPoints->dimensionality, epanechnikovFactor);
    }
    density *= parzenFactor;

    free(scaledPattern);
    return density;
}

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