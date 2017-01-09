//
// Created by Laura Baakman on 09/01/2017.
//

#include "kde.h"

static char kernels_standardGaussian_docstring[] = "Estimate the densities with Parzen.";
static PyObject * kdeParzenMultiPattern(PyObject *self, PyObject *args){
    PyObject* inPatterns = NULL;
    PyObject* inDataPoints = NULL;
    PyObject* outDensities = NULL;

    PyArrayObject* patterns = NULL;
    PyArrayObject* dataPoints = NULL;
    PyArrayObject* densities = NULL;

    if (!PyArg_ParseTuple(args, "OOO", &inPatterns, &inDataPoints, &outDensities)) goto fail;

    patterns = (PyArrayObject *)PyArray_FROM_OTF(inPatterns, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (patterns == NULL) goto fail;

    dataPoints = (PyArrayObject *)PyArray_FROM_OTF(inDataPoints, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (dataPoints == NULL) goto fail;

    densities = (PyArrayObject *)PyArray_FROM_OTF(outDensities, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);
    if (densities == NULL) goto fail;

    double* densities_data = (double *)PyArray_DATA(densities);

    int num_patterns = (int)PyArray_DIM(patterns, 0);
    int dim_patterns = (int)PyArray_DIM(patterns, 1);

    int pattern_stride = (int)PyArray_STRIDE (patterns, 0) / (int)PyArray_ITEMSIZE(patterns);

    double* current_pattern = (double*)PyArray_DATA(patterns);

    for(int j = 0; j < num_patterns; j++)
    {
        densities_data[j] = parzen(current_pattern, dim_patterns, dataPoints);
        current_pattern += pattern_stride;
    }

    /* Clean up Memory */
    Py_DECREF(patterns);
    Py_XDECREF(dataPoints);
    Py_XDECREF(densities);

    /* Create return object */
    Py_INCREF(Py_None);
    return Py_None;

    fail:
    Py_XDECREF(patterns);
    Py_XDECREF(dataPoints);
    Py_XDECREF(densities);
    return NULL;
}


static PyMethodDef method_table[] = {
        {"parzen_multi_pattern",     kdeParzenMultiPattern, METH_VARARGS,   kernels_standardGaussian_docstring},
        /* Sentinel */
        {NULL,                              NULL,                                   0,              NULL}
};

static struct PyModuleDef kernelModule = {
        PyModuleDef_HEAD_INIT, "_kernels",
        "C implementation of some kernels.",
        -1, method_table
};

PyMODINIT_FUNC PyInit__kde(void) {
    PyObject *module = PyModule_Create(&kernelModule);

    if(!module) return NULL;
    import_array();

    return module;
}