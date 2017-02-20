//
// Created by Laura Baakman on 16/02/2017.
//

#ifndef KERNELS_UTILSMODULE_H_H
#define KERNELS_UTILSMODULE_H_H

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "../utils.h"
#include "distancematrix.h"
#include "knn.h"
#include "covariancematrix.h"

Array pyObjectToArray(PyObject *pythonObject, int requirements);

#endif //KERNELS_UTILSMODULE_H_H
