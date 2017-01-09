//
// Created by Laura Baakman on 19/12/2016.
//

#include "kernels.h"

static char kernels_standardGaussian_docstring[] = "Evaluate the Standard Gaussian (zero vector mean and identity covariance matrix) for each row in the input matrix.";
static PyObject * standard_gaussian_multi_pattern(PyObject *self, PyObject *args){
    PyObject* inPatterns = NULL;
    PyObject* outDensities = NULL;

    PyArrayObject* patterns = NULL;
    PyArrayObject* densities = NULL;

    if (!PyArg_ParseTuple(args, "OO", &inPatterns, &outDensities)) goto fail;

    patterns = (PyArrayObject *)PyArray_FROM_OTF(inPatterns, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (patterns == NULL) goto fail;

    densities = (PyArrayObject *)PyArray_FROM_OTF(outDensities, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);
    if (densities == NULL) goto fail;

    double* densities_data = (double *)PyArray_DATA(densities);

    int num_patterns = (int)PyArray_DIM(patterns, 0);
    int dim_patterns = (int)PyArray_DIM(patterns, 1);

    int pattern_stride = (int)PyArray_STRIDE (patterns, 0) / (int)PyArray_ITEMSIZE(patterns);

    double* current_pattern = (double*)PyArray_DATA(patterns);

    double factor = standardGaussianFactor(dim_patterns);

    for(int j = 0; j < num_patterns; j++)
    {
        densities_data[j] = standardGaussian(current_pattern, dim_patterns, factor);
        current_pattern += pattern_stride;
    }

    /* Clean up Memory */
    Py_DECREF(patterns);
    Py_XDECREF(densities);

    /* Create return object */
    Py_INCREF(Py_None);
    return Py_None;

    fail:
    Py_XDECREF(patterns);
    Py_XDECREF(densities);
    return NULL;
}

static PyObject * standard_gaussian_single_pattern(PyObject *self, PyObject *args){
    PyObject* inPatterns = NULL;
    PyArrayObject* pattern = NULL;

    if (!PyArg_ParseTuple(args, "O", &inPatterns)) goto fail;

    pattern = (PyArrayObject *)PyArray_FROM_OTF(inPatterns, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (pattern == NULL) goto fail;

    int dim_pattern = (int)PyArray_DIM(pattern, 0);

    double* pattern_data = (double*)PyArray_DATA(pattern);
    double factor = standardGaussianFactor(dim_pattern);
    double density = standardGaussian(pattern_data, dim_pattern, factor);

    /* Clean up Memory */
    Py_DECREF(pattern);

    /* Create return object */
    PyObject *returnObject = Py_BuildValue("d", density);
    return returnObject;

    fail:
    Py_XDECREF(pattern);
    return NULL;
}

static PyMethodDef method_table[] = {
        {"standard_gaussian_multi_pattern",     standard_gaussian_multi_pattern, METH_VARARGS,   kernels_standardGaussian_docstring},
        {"standard_gaussian_single_pattern",    standard_gaussian_single_pattern,   METH_VARARGS,   kernels_standardGaussian_docstring},
        /* Sentinel */
        {NULL,                              NULL,                                   0,              NULL}
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