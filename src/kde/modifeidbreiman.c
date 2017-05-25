#include <gsl/gsl_vector_double.h>
#include "modifeidbreiman.h"

double modifiedBreimanFinalDensity(double *pattern, Array *dataPoints,
                                   double globalBandwidth, Array *localBandwidths,
                                   SymmetricKernelDensityFunction kernel){
    double* currentDataPoint = dataPoints->data;

    double* scaledPattern = (double *)malloc(sizeof(double) * dataPoints->dimensionality);

    double density = 0;
    double factor, bandwidth;

    gsl_vector_view currentPattern;

    for(int i = 0;
            i < dataPoints->length;
            ++i, currentDataPoint+= dataPoints->rowStride){

        bandwidth = globalBandwidth * localBandwidths->data[i];
        factor = pow(bandwidth, -1 * dataPoints->dimensionality);
        scaledPattern = scalePattern(pattern, currentDataPoint, scaledPattern, dataPoints->dimensionality, bandwidth);
        currentPattern = gsl_vector_view_array(scaledPattern, (size_t)dataPoints->dimensionality);
        density += (factor * kernel(&currentPattern.vector));
    }
    density /= dataPoints->length;

    free(scaledPattern);
    return density;
}