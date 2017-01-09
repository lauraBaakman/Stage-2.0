//
// Created by Laura Baakman on 09/01/2017.
//

#ifndef KERNELS_DENSITYFUNCTIONS_H
#define KERNELS_DENSITYFUNCTIONS_H

#include <printf.h>
#include <math.h>

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

double parzen(double* pattern, int dimensionality, PyArrayObject *dataPoints, double windowWidth);

#endif //KERNELS_DENSITYFUNCTIONS_H
