//
// Created by Laura Baakman on 09/01/2017.
//

#ifndef KDE_H
#define KDE_H

#include <math.h>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "parzen.h"
#include "breiman.h"
#include "utils.h"
#include "kernels/densityFunctions.h"

Array pyObjectToArray(PyObject *pythonObject, int requirements);

#endif //KDE_H
