//
// Created by Laura Baakman on 19/12/2016.
//

#ifndef KERNELS_STANDARDGAUSSIAN_H
#define KERNELS_STANDARDGAUSSIAN_H

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>

static char kernels_standardGaussian_docstring[] = "Evaluate the Standard Gaussian (zero vector mean and identity covariance matrix) for each row in the input matrix.";
PyObject * kernels_standardGaussian(PyObject *self, PyObject * args);

#endif //KERNELS_STANDARDGAUSSIAN_H
