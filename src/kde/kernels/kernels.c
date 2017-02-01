//
// Created by Laura Baakman on 19/12/2016.
//

#include "kernels.h"
#include "../../../../../.virtualenvs/stage/include/python3.5m/object.h"

static char kernels_standardGaussian_docstring[] = "Evaluate the Standard Gaussian (zero vector mean and identity covariance matrix) for each row in the input matrix.";
static PyObject * standard_gaussian_multi_pattern(PyObject *self, PyObject *args){
    PyObject* inPatterns = NULL;
    PyObject* outDensities = NULL;

    if (!PyArg_ParseTuple(args, "OO", &inPatterns, &outDensities)) return NULL;

    Array patterns = pyObjectToArray(inPatterns, NPY_ARRAY_IN_ARRAY);
    Array densities = pyObjectToArray(outDensities, NPY_ARRAY_OUT_ARRAY);

    double* current_pattern = patterns.data;

    double factor = standardGaussianFactor(patterns.dimensionality);

    for(
            int j = 0;
            j < patterns.length;
            j++, current_pattern += patterns.stride)
    {
        densities.data[j] = standardGaussian(current_pattern, patterns.dimensionality, factor);
    }
    /* Create return object */
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * standard_gaussian_single_pattern(PyObject *self, PyObject *args){
    PyObject* inPatterns = NULL;

    if (!PyArg_ParseTuple(args, "O", &inPatterns)) return NULL;

    Array pattern = pyObjectToArray(inPatterns, NPY_ARRAY_IN_ARRAY);

    double factor = standardGaussianFactor(pattern.dimensionality);
    double density = standardGaussian(pattern.data, pattern.dimensionality, factor);

    /* Create return object */
    PyObject *returnObject = Py_BuildValue("d", density);
    return returnObject;
}

static char kernels_epanechnikov_docstring[] = "Evaluate the Epanechnikov kernel for each row in the input matrix.";
static PyObject * epanechnikov_single_pattern(PyObject *self, PyObject *args){
    PyObject* inPatterns = NULL;

    if (!PyArg_ParseTuple(args, "O", &inPatterns)) return NULL;

    Array pattern = pyObjectToArray(inPatterns, NPY_ARRAY_IN_ARRAY);

    double denominator = epanechnikovDenominator(pattern.dimensionality);
    double density = epanechnikov(pattern.data, pattern.dimensionality, denominator);

    /* Create return object */
    PyObject *returnObject = Py_BuildValue("d", density);
    return returnObject;
}

static PyObject * epanechnikov_multi_pattern(PyObject *self, PyObject *args){
    PyObject* inPatterns = NULL;
    PyObject* outDensities = NULL;

    if (!PyArg_ParseTuple(args, "OO", &inPatterns, &outDensities)) return NULL;

    Array patterns = pyObjectToArray(inPatterns, NPY_ARRAY_IN_ARRAY);
    Array densities = pyObjectToArray(outDensities, NPY_ARRAY_OUT_ARRAY);

    double denominator = epanechnikovDenominator(patterns.dimensionality);

    double* currentPattern = patterns.data;

    for (int i = 0;
         i < patterns.length;
         ++i, currentPattern += patterns.stride)
    {
        densities.data[i] = epanechnikov(currentPattern, patterns.dimensionality, denominator);
    }

    /* Create return object */
    Py_INCREF(Py_None);
    return Py_None;
}

static char kernels_testKernel_docstring[] = "Evaluate the TestKernel for each row in the input matrix. This kernel returns the absolute value of the mean of the elements of the patterns.";
static PyObject * testKernel_single_pattern(PyObject *self, PyObject *args){
    PyObject* inPatterns = NULL;

    if (!PyArg_ParseTuple(args, "O", &inPatterns)) return NULL;

    Array pattern = pyObjectToArray(inPatterns, NPY_ARRAY_IN_ARRAY);

    double factor = testKernelFactor(pattern.dimensionality);
    double density = testKernel(pattern.data, pattern.dimensionality, factor);

    /* Create return object */
    PyObject *returnObject = Py_BuildValue("d", density);
    return returnObject;
}

static PyObject * testKernel_multi_pattern(PyObject *self, PyObject *args){
    PyObject* inPatterns = NULL;
    PyObject* outDensities = NULL;

    if (!PyArg_ParseTuple(args, "OO", &inPatterns, &outDensities)) return NULL;

    Array patterns = pyObjectToArray(inPatterns, NPY_ARRAY_IN_ARRAY);
    Array densities = pyObjectToArray(outDensities, NPY_ARRAY_OUT_ARRAY);

    double factor = testKernelFactor(patterns.dimensionality);

    double* currentPattern = patterns.data;

    for (int i = 0;
         i < patterns.length;
         ++i, currentPattern += patterns.stride)
    {
        densities.data[i] = testKernel(currentPattern, patterns.dimensionality, factor);
    }

    /* Create return object */
    Py_INCREF(Py_None);
    return Py_None;
}


Array pyObjectToArray(PyObject *pythonObject, int requirements){
    PyArrayObject* arrayObject = NULL;
    arrayObject = (PyArrayObject *)PyArray_FROM_OTF(pythonObject, NPY_DOUBLE, requirements);
    if (arrayObject == NULL){
        fprintf(stderr, "Error converting PyObject to PyArrayObject\n");
        exit(-1);
    }
    Array array = buildArrayFromPyArray(arrayObject);
    Py_XDECREF(arrayObject);
    return array;
}

static PyMethodDef method_table[] = {
        {"standard_gaussian_multi_pattern",     standard_gaussian_multi_pattern,    METH_VARARGS,   kernels_standardGaussian_docstring},
        {"standard_gaussian_single_pattern",    standard_gaussian_single_pattern,   METH_VARARGS,   kernels_standardGaussian_docstring},
        {"epanechnikov_single_pattern",         epanechnikov_single_pattern,        METH_VARARGS,   kernels_epanechnikov_docstring},
        {"epanechnikov_multi_pattern",          epanechnikov_multi_pattern,         METH_VARARGS,   kernels_epanechnikov_docstring},
        {"test_kernel_single_pattern",          testKernel_single_pattern,          METH_VARARGS,   kernels_testKernel_docstring},
        {"test_kernel_multi_pattern",           testKernel_multi_pattern,           METH_VARARGS,   kernels_testKernel_docstring},
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