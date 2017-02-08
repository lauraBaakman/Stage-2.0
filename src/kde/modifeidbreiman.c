#include "modifeidbreiman.h"

double modifiedBreimanFinalDensity(double *pattern, Array *dataPoints,
                                   double globalBandwidth, Array *localBandwidths,
                                   double kernelConstant, KernelDensityFunction kernel){
    double* currentDataPoint = dataPoints->data;
    double* currentLocalBandWidth = localBandwidths->data;

    double* scaledPattern = (double *)malloc(sizeof(double) * dataPoints->dimensionality);

    double density = 0;
    double factor, bandwidth;

    for(int i = 0;
            i < dataPoints->length;
            ++i, currentDataPoint+= dataPoints->stride, currentLocalBandWidth+= localBandwidths->stride){
        bandwidth = globalBandwidth * (*currentLocalBandWidth);
        factor = pow(bandwidth, -1 * dataPoints->dimensionality);
        scaledPattern = scalePattern(pattern, currentDataPoint, scaledPattern, dataPoints->dimensionality, bandwidth);
        density += (factor * kernel(scaledPattern, dataPoints->dimensionality, kernelConstant));
    }
    density /= dataPoints->length;

    free(scaledPattern);
    return density;
}