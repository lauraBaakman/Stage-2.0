//
// Created by Laura Baakman on 09/01/2017.
//

#include "parzen.h"

double parzen(double* pattern, int dimensionality, PyArrayObject* dataPoints){
    int numDataPoints = (int)PyArray_DIM(dataPoints, 0);

    double* dataDataPoints = (double *)PyArray_DATA(dataPoints);

    int strideDataPoints = (int)PyArray_STRIDE (dataPoints, 0) / (int)PyArray_ITEMSIZE(dataPoints);

    double* currentDataPoint = dataDataPoints;

    for (int i = 0; i < numDataPoints; ++i) {
        for(int j = 0; j < dimensionality; j++) {
            printf("%f ", currentDataPoint[j]);
        }
        printf("\n");
        currentDataPoint += strideDataPoints;
    }
    printf("\n");
    return 42.0;
}

