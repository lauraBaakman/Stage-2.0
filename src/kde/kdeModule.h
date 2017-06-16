//
// Created by Laura Baakman on 09/01/2017.
//

#ifndef KDE_MODULE_H
#define KDE_MODULE_H

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector_double.h>

#include "parzen.h"
#include "mbe.h"
#include "sambe.h"
#include "kernels/kernels.h"

gsl_matrix_view pyObjectToGSLMatrixView(PyObject *pythonObject, int requirements);
gsl_vector_view pyObjectToGSLVectorView(PyObject *pythonObject, int requirements);

Kernel selectKernel(KernelType type);

#endif //KDE_MODULE_H
