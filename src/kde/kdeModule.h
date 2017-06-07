//
// Created by Laura Baakman on 09/01/2017.
//

#ifndef KDE_MODULE_H
#define KDE_MODULE_H

#include <math.h>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "parzen.h"
#include "mbe.h"
#include "sambe.h"
#include "sambe.ih" // Shouldn't be included anywhere, but the locals needed to be exposed somewhere for easy testing.
#include "utils.h"
#include "kernels/kernels.h"

Array pyObjectToArray(PyObject *pythonObject, int requirements);
gsl_matrix_view pyObjectToGSLMatrixView(PyObject *pythonObject, int requirements);
gsl_vector_view pyObjectToGSLVectorView(PyObject *pythonObject, int requirements);

Kernel selectKernel(KernelType type);

#endif //KDE_MODULE_H
