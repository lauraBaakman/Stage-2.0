//
// Created by Laura Baakman on 19/12/2016.
//

#ifndef KERNELS_KERNELS_H
#define KERNELS_KERNELS_H

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "kernels.h"
#include "../utils.h"

Array pyObjectToArray(PyObject *pythonObject, int requirements);

#endif //KERNELS_KERNELS_H
