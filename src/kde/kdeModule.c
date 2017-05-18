//
// Created by Laura Baakman on 09/01/2017.
//

#include "kdeModule.h"

static char kde_parzen_docstring[] = "Estimate densities with Parzen.";
static PyObject * kdeParzen(PyObject *self, PyObject *args){
    PyObject* inPatterns = NULL;
    PyObject* inDataPoints = NULL;
    PyObject* outDensities = NULL;

    double inWindowWidth;
    KernelType inKernelType;

    if (!PyArg_ParseTuple(args, "OOdiO",
                          &inPatterns,
                          &inDataPoints,
                          &inWindowWidth,
                          &inKernelType,
                          &outDensities)) return NULL;

    Array patterns = pyObjectToArray(inPatterns, NPY_ARRAY_IN_ARRAY);
    Array dataPoints = pyObjectToArray(inDataPoints, NPY_ARRAY_IN_ARRAY);
    Array densities = pyObjectToArray(outDensities, NPY_ARRAY_OUT_ARRAY);

    double parzenFactor = 1.0 / (dataPoints.length * pow(inWindowWidth, patterns.dimensionality));

    SymmetricKernel kernel = selectSymmetricKernel(inKernelType);
    double kernelConstant = kernel.factorFunction(dataPoints.dimensionality);

    double* current_pattern = patterns.data;

    for(int j = 0;
        j < patterns.length;
        j++, current_pattern += patterns.rowStride)
    {
        densities.data[j] = parzen(current_pattern, &dataPoints,
                                   inWindowWidth, parzenFactor,
                                   kernel.densityFunction, kernelConstant);
    }

    /* Create return object */
    Py_INCREF(Py_None);
    return Py_None;
}

static char kde_breiman_docstring[] = "Perform the final estimation step of the Modified breiman estimator.";
static PyObject *kde_modified_breiman(PyObject *self, PyObject *args){
    PyObject* inPatterns = NULL;
    PyObject* inDataPoints = NULL;
    PyObject* inLocalBandwidths = NULL;
    PyObject* outDensities = NULL;


    double globalBandwidth;
    KernelType inKernelType;

    if (!PyArg_ParseTuple(args, "OOdOiO",
                          &inPatterns,
                          &inDataPoints,
                          &globalBandwidth,
                          &inLocalBandwidths,
                          &inKernelType,
                          &outDensities)) return NULL;

    Array patterns = pyObjectToArray(inPatterns, NPY_ARRAY_IN_ARRAY);
    Array dataPoints = pyObjectToArray(inDataPoints, NPY_ARRAY_IN_ARRAY);
    Array localBandwidths = pyObjectToArray(inLocalBandwidths, NPY_ARRAY_IN_ARRAY);
    Array densities = pyObjectToArray(outDensities, NPY_ARRAY_OUT_ARRAY);

    SymmetricKernel kernel = selectSymmetricKernel(inKernelType);
    double kernelConstant = kernel.factorFunction(dataPoints.dimensionality);

    double* current_pattern = patterns.data;

    for(int j = 0;
        j < patterns.length;
        j++, current_pattern += patterns.rowStride)
    {
        densities.data[j] = modifiedBreimanFinalDensity(current_pattern, &dataPoints,
                                                        globalBandwidth, &localBandwidths,
                                                        kernelConstant, kernel.densityFunction);
    }

    /* Create return object */
    Py_INCREF(Py_None);
    return Py_None;
}

static char kde_sambe_docstring[] = "Perform the final estimation step of the Shape Adaptive Modified breiman estimator.";
static PyObject *kde_shape_adaptive_mbe(PyObject *self, PyObject *args){

    /* Parse Arguments */
    PyObject* inPatterns = NULL;
    PyObject* inLocalBandwidths = NULL;
    PyObject* outDensities = NULL;
    KernelType kernelType;
    int k;
    double globalBandwidth;


    if (!PyArg_ParseTuple(args, "OiidOO",
                          &inPatterns,
                          &kernelType,
                          &k,
                          &globalBandwidth,
                          &inLocalBandwidths,
                          &outDensities)) return NULL;

    Array patterns = pyObjectToArray(inPatterns, NPY_ARRAY_IN_ARRAY);
    Array localBandwidths = pyObjectToArray(inLocalBandwidths, NPY_ARRAY_IN_ARRAY);
    Array densities = pyObjectToArray(outDensities, NPY_ARRAY_OUT_ARRAY);

    double* current_pattern = patterns.data;

    ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(kernelType);

    /* Do computations */
    for(int j = 0;
        j < patterns.length;
        j++, current_pattern += patterns.rowStride)
    {
        densities.data[j] = sambeFinalDensity(current_pattern, &patterns, globalBandwidth, kernel);
    }

    /* Free memory */

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
    Array array = arrayBuildFromPyArray(arrayObject);
    Py_XDECREF(arrayObject);
    return array;
}

static PyMethodDef method_table[] = {
        {"parzen",              kdeParzen,                          METH_VARARGS,   kde_parzen_docstring},
        {"modified_breiman",    kde_modified_breiman,               METH_VARARGS,   kde_breiman_docstring},
        {"shape_adaptive_mbe",  kde_shape_adaptive_mbe,             METH_VARARGS,   kde_sambe_docstring},
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