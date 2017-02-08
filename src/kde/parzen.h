//
// Created by Laura Baakman on 09/01/2017.
//

#ifndef PARZEN_H
#define PARZEN_H

#include <printf.h>
#include <math.h>

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "kernels/densityFunctions.h"
#include "utils.h"

double parzen_gaussian(double *pattern, Array *dataPoints, double windowWidth, double parzenFactor,
                       double gaussianFactor);
double parzen_epanechnikov(double *pattern, Array *dataPoints, double windowWidth, double parzenFactor,
                       double epanechnikovFactor);

//'Private stuff', didn't feel like messing around with internal headers
double* scalePattern(double* pattern, double* dataPoint, double* scaledPattern, int dimensionality, double windowWidth);
#endif //PARZEN_H
