//
// Created by Laura Baakman on 09/01/2017.
//

#include "parzen.ih"

double parzen(double *pattern, Array *dataPoints, double windowWidth, double parzenFactor,
              SymmetricKernelDensityFunction kernel, double  kernelConstant){
    double* currentDataPoint = dataPoints->data;
    double density = 0;

    double* scaledPattern = (double *)malloc(sizeof(double) * dataPoints->dimensionality);

    for (int i = 0;
         i < dataPoints->length;
         ++i, currentDataPoint += dataPoints->rowStride)
    {
        scaledPattern = scalePattern(pattern, currentDataPoint, scaledPattern, dataPoints->dimensionality, windowWidth);
        density += kernel(scaledPattern, dataPoints->dimensionality, kernelConstant);
    }
    density *= parzenFactor;

    free(scaledPattern);
    return density;
}

