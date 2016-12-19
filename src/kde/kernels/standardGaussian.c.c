//
// Created by Laura Baakman on 19/12/2016.
//

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>
#include "standardGaussian.c.h"

static char standardGaussian_docstring[] = "Evaluate the Standard Gaussian (zero vector mean and identity covariance matrix) for each row in the input matrix.";
static PyObject * kernels_standardGaussian(PyObject *self, PyObject * args){

    // Build return value   
    PyObject * returnObject = Py_BuildValue("d", 42.0);
    return returnObject;
}

static PyMethodDef method_table[] = {
        {"standard_gaussian",   kernels_standardGaussian,   METH_VARARGS,   standardGaussian_docstring},
        /* Sentinel */
        {NULL,                  NULL,                       0,              NULL}
};

static struct PyModuleDef kernelModule = {
        PyModuleDef_HEAD_INIT, "_kernels",
        "C implementation of some kernels.",
        -1, method_table
};

PyMODINIT_FUNC PyInit__kernels(void) {
    PyObject *module = PyModule_Create(&kernelModule);

    if(!module) return NULL;
    import_array();

    return module;
}