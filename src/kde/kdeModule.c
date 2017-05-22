//
// Created by Laura Baakman on 09/01/2017.
//

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector_double.h>
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
    PyObject* inXis = NULL;
    PyObject* inXs = NULL;
    PyObject* inLocalBandwidths = NULL;
    PyObject* outDensities = NULL;
    KernelType kernelType;
    int k;
    double globalBandwidth;


    if (!PyArg_ParseTuple(args, "OOiidOO",
                          &inXis,
                          &inXs,
                          &kernelType,
                          &k,
                          &globalBandwidth,
                          &inLocalBandwidths,
                          &outDensities)) return NULL;

    gsl_matrix_view xis = pyObjectToGSLMatrixView(inXis, NPY_ARRAY_IN_ARRAY);
    gsl_matrix_view xs = pyObjectToGSLMatrixView(inXs, NPY_ARRAY_IN_ARRAY);
    gsl_vector_view localBandwidths = pyObjectToGSLVectorView(inLocalBandwidths, NPY_ARRAY_IN_ARRAY);
    gsl_vector_view densities = pyObjectToGSLVectorView(outDensities, NPY_ARRAY_OUT_ARRAY);

    double density = 42.0;

    /* Do computations */
    for(size_t patternIdx = 0; patternIdx < xs.matrix.size1; patternIdx++){
        gsl_vector_set(&densities.vector, patternIdx, density);
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

gsl_vector_view pyObjectToGSLVectorView(PyObject *pythonObject, int requirements) {
    PyArrayObject* arrayObject = NULL;
    arrayObject = (PyArrayObject *)PyArray_FROM_OTF(pythonObject, NPY_DOUBLE, requirements);
    if (arrayObject == NULL){
        fprintf(stderr, "Error converting PyObject to PyArrayObject\n");
        exit(-1);
    }
    double* data = (double *)PyArray_DATA(arrayObject);
    size_t length = (size_t) PyArray_DIM(arrayObject, 0);

    Py_XDECREF(arrayObject);
    return gsl_vector_view_array(data, length);
}

gsl_matrix_view pyObjectToGSLMatrixView(PyObject *pythonObject, int requirements) {
    PyArrayObject* arrayObject = NULL;
    arrayObject = (PyArrayObject *)PyArray_FROM_OTF(pythonObject, NPY_DOUBLE, requirements);
    if (arrayObject == NULL){
        fprintf(stderr, "Error converting PyObject to PyArrayObject\n");
        exit(-1);
    }
    double* data = (double *)PyArray_DATA(arrayObject);
    size_t num_rows = (size_t) PyArray_DIM(arrayObject, 0);
    size_t num_cols = (size_t) PyArray_DIM(arrayObject, 1);

    Py_XDECREF(arrayObject);
    return gsl_matrix_view_array(data, num_rows, num_cols);
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