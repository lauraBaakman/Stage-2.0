#ifndef KERNELS_KERNELS_MODULE_H
#define KERNELS_KERNELS_MODULE_H

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "kernels.h"
#include "../utils.h"

Array pyObjectToArray(PyObject *pythonObject, int requirements);

static void multi_pattern_symmetric(SymmetricKernel kernel, Array* patterns, Array* densities);
static void multi_pattern_asymmetric(ASymmetricKernel kernel, Array* patterns, Array* densities);

#endif //KERNELS_KERNELS_MODULE_H
