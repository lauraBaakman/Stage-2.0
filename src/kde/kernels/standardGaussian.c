//
// Created by Laura Baakman on 19/12/2016.
//


#include "standardGaussian.h"

PyObject * kernels_standardGaussian(PyObject *self, PyObject * args){

    // Build return value
    PyObject * returnObject = Py_BuildValue("d", 42.0);
    return returnObject;
}