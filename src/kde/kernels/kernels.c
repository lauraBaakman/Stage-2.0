//
// Created by Laura Baakman on 19/12/2016.
//

#include "kernels.h"

static PyMethodDef method_table[] = {
        {"standard_gaussian",   kernels_standardGaussian,   METH_VARARGS,   kernels_standardGaussian_docstring},
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