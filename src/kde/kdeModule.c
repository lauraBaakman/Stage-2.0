//
// Created by Laura Baakman on 09/01/2017.
//

#include "kdeModule.h"

static char kde_parzen_docstring[] = "Estimate densities with Parzen.";
static PyObject * kdeParzen(PyObject *self, PyObject *args){
    PyObject* inPatterns = NULL;
    PyObject* inDataPoints = NULL;
    PyObject* outDensities = NULL;

    double windowWidth;
    KernelType kernelType;

    if (!PyArg_ParseTuple(args, "OOdiO",
                          &inPatterns,
                          &inDataPoints,
                          &windowWidth,
                          &kernelType,
                          &outDensities)) return NULL;

    Array patterns = pyObjectToArray(inPatterns, NPY_ARRAY_IN_ARRAY);
    Array dataPoints = pyObjectToArray(inDataPoints, NPY_ARRAY_IN_ARRAY);
    Array densities = pyObjectToArray(outDensities, NPY_ARRAY_OUT_ARRAY);

    double parzenFactor = 1.0 / (dataPoints.length * pow(windowWidth, patterns.dimensionality));

    Kernel kernel = selectKernel(kernelType);
    double kernelConstant = kernel.factorFunction(dataPoints.dimensionality);

    double* current_pattern = patterns.data;

    for(int j = 0;
        j < patterns.length;
        j++, current_pattern += patterns.stride)
    {
        densities.data[j] = parzen(current_pattern, &dataPoints,
                                   windowWidth, parzenFactor,
                                   kernel.densityFunction, kernelConstant);
    }

    /* Create return object */
    Py_INCREF(Py_None);
    return Py_None;
}

static char kde_breiman_epanechnikov_docstring[] = "Estimate the densities with Parzen with the Epanechnikov kernel.";
static PyObject *kde_modifeid_breiman(PyObject *self, PyObject *args){
    PyObject* inPatterns = NULL;
    PyObject* inDataPoints = NULL;
    PyObject* inLocalBandwidths = NULL;
    PyObject* outDensities = NULL;


    double globalBandwidth;

    if (!PyArg_ParseTuple(args, "OOdOO",
                          &inPatterns,
                          &inDataPoints,
                          &globalBandwidth,
                          &inLocalBandwidths,
                          &outDensities)) return NULL;

    Array patterns = pyObjectToArray(inPatterns, NPY_ARRAY_IN_ARRAY);
    Array dataPoints = pyObjectToArray(inDataPoints, NPY_ARRAY_IN_ARRAY);
    Array localBandwidths = pyObjectToArray(inLocalBandwidths, NPY_ARRAY_IN_ARRAY);
    Array densities = pyObjectToArray(outDensities, NPY_ARRAY_OUT_ARRAY);

    double parzenFactor = 1.0 / (dataPoints.length * pow(globalBandwidth, patterns.dimensionality));

    Kernel kernel = selectKernel(EPANECHNIKOV);
    double kernelConstant = kernel.factorFunction(dataPoints.dimensionality);

    double* current_pattern = patterns.data;

    for(int j = 0;
        j < patterns.length;
        j++, current_pattern += patterns.stride)
    {
        densities.data[j] = modifeidBreimenFinalDensity(current_pattern, &dataPoints,
                                                        globalBandwidth, &localBandwidths, parzenFactor,
                                                        kernelConstant, kernel.densityFunction);
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
        {"parzen",                      kdeParzen,                  METH_VARARGS,   kde_parzen_docstring},
        {"breiman_epanechnikov",        kde_modifeid_breiman,     METH_VARARGS,   kde_breiman_epanechnikov_docstring},
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