#ifndef KERNELS_KERNELS_MODULE_H
#define KERNELS_KERNELS_MODULE_H

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "kernels.h"
#include "../utils.h"

Array pyObjectToArray(PyObject *pythonObject, int requirements);

static PyObject * multi_pattern(PyObject *args, KernelType kernelType);

#endif //KERNELS_KERNELS_MODULE_H
