//
// Created by Laura Baakman on 09/01/2017.
//

#include "kde.h"

static char kde_parzen_standardGaussian_docstring[] = "Estimate the densities with Parzen with a Gaussian kernel.";
static PyObject * kdeParzenStandardGaussian(PyObject *self, PyObject *args){
    PyObject* inPatterns = NULL;
    PyObject* inDataPoints = NULL;
    PyObject* outDensities = NULL;

    double windowWidth;

    if (!PyArg_ParseTuple(args, "OOdO",
                          &inPatterns,
                          &inDataPoints,
                          &windowWidth,
                          &outDensities)) return NULL;

    Array patterns = pyObjectToArray(inPatterns, NPY_ARRAY_IN_ARRAY);
    Array dataPoints = pyObjectToArray(inDataPoints, NPY_ARRAY_IN_ARRAY);
    Array densities = pyObjectToArray(outDensities, NPY_ARRAY_OUT_ARRAY);

    double parzenFactor = 1.0 / (dataPoints.length * pow(windowWidth, patterns.dimensionality));
    double gaussianFactor = standardGaussianFactor(dataPoints.dimensionality);

    double* current_pattern = patterns.data;

    for(int j = 0;
        j < patterns.length;
        j++, current_pattern += patterns.stride)
    {
        densities.data[j] = parzen_gaussian(current_pattern, &dataPoints, windowWidth, parzenFactor, gaussianFactor);
    }

    /* Create return object */
    Py_INCREF(Py_None);
    return Py_None;
}

static char kde_parzen_epanechnikov_docstring[] = "Estimate the densities with Parzen with the Epanechnikov kernel.";
static PyObject * kdeParzenEpanechnikov(PyObject *self, PyObject *args){
    PyObject* inPatterns = NULL;
    PyObject* inDataPoints = NULL;
    PyObject* outDensities = NULL;

    double windowWidth;

    if (!PyArg_ParseTuple(args, "OOdO",
                          &inPatterns,
                          &inDataPoints,
                          &windowWidth,
                          &outDensities)) return NULL;

    Array patterns = pyObjectToArray(inPatterns, NPY_ARRAY_IN_ARRAY);
    Array dataPoints = pyObjectToArray(inDataPoints, NPY_ARRAY_IN_ARRAY);
    Array densities = pyObjectToArray(outDensities, NPY_ARRAY_OUT_ARRAY);

    double parzenFactor = 1.0 / (dataPoints.length * pow(windowWidth, patterns.dimensionality));
    double epanechnikovFactor = epanechnikovDenominator(dataPoints.dimensionality);

    double* current_pattern = patterns.data;

    for(int j = 0;
        j < patterns.length;
        j++, current_pattern += patterns.stride)
    {
        densities.data[j] = parzen_epanechnikov(current_pattern, &dataPoints, windowWidth, parzenFactor, epanechnikovFactor);
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
        {"parzen_standard_gaussian",    kdeParzenStandardGaussian,  METH_VARARGS,   kde_parzen_standardGaussian_docstring},
        {"parzen_epanechnikov",         kdeParzenEpanechnikov,      METH_VARARGS,   kde_parzen_epanechnikov_docstring},
        /* Sentinel */
        {NULL,                              NULL,                                   0,              NULL}
};

static struct PyModuleDef kernelModule = {
        PyModuleDef_HEAD_INIT, "_kernels",
        "C implementation of some kernel density estimation methods.",
        -1, method_table
};

PyMODINIT_FUNC PyInit__kde(void) {
    PyObject *module = PyModule_Create(&kernelModule);

    if(!module) return NULL;
    import_array();

    return module;
}