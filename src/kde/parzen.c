#include <gsl/gsl_vector_double.h>
#include "parzen.ih"

double parzen(double *pattern, Array *dataPoints, double windowWidth, double parzenFactor,
              SymmetricKernelDensityFunction kernel, double  kernelConstant){
    double* currentDataPoint = dataPoints->data;
    double density = 0;

    double* scaledPattern = (double *)malloc(sizeof(double) * dataPoints->dimensionality);
    gsl_vector_view currentPattern;

    for (int i = 0;
         i < dataPoints->length;
         ++i, currentDataPoint += dataPoints->rowStride)
    {
        scaledPattern = scalePattern(pattern, currentDataPoint, scaledPattern, dataPoints->dimensionality, windowWidth);
        currentPattern = gsl_vector_view_array(scaledPattern, (size_t) dataPoints->dimensionality);
        density += kernel(&currentPattern.vector);
    }
    density *= parzenFactor;

    free(scaledPattern);
    return density;
}

