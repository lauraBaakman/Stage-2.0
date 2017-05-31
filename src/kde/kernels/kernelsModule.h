#ifndef KERNELS_KERNELS_MODULE_H
#define KERNELS_KERNELS_MODULE_H

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <gsl/gsl_matrix.h>

#include "kernels.h"
#include "../utils.h"
#include "../utils/gsl_utils.h"

Array pyObjectToArray(PyObject *pythonObject, int requirements);
gsl_matrix_view pyObjectToGSLMatrixView(PyObject *pythonObject, int requirements);
gsl_matrix* pyObjectToGSLMatrix(PyObject *pythonObject, int requirements);
gsl_vector_view pyObjectToGSLVectorView(PyObject *pythonObject, int requirements);

static PyObject * multi_pattern_symmetric(PyObject *args, KernelType kernelType);
static PyObject* single_pattern_symmetric(PyObject *args, KernelType kernelType);

static PyObject *single_pattern_shapeadaptive(PyObject *args, KernelType kernelType);

#endif //KERNELS_KERNELS_MODULE_H
