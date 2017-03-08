#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector_double.h>
#include "kernelsModule.h"
#include "../../../../../.virtualenvs/stage/include/python3.5m/object.h"

PyObject *multi_pattern_symmetric(PyObject *args, KernelType kernelType) {
    /* Parse input data */

    PyObject* inPatterns = NULL;
    PyObject* outDensities = NULL;

    if (!PyArg_ParseTuple(args, "OO", &inPatterns, &outDensities)) return NULL;

    Array patterns = pyObjectToArray(inPatterns, NPY_ARRAY_IN_ARRAY);
    Array densities = pyObjectToArray(outDensities, NPY_ARRAY_OUT_ARRAY);

    /* Do computations */
    SymmetricKernel kernel = selectSymmetricKernel(kernelType);

    double* current_pattern = patterns.data;
    double kernelConstant = kernel.factorFunction(patterns.dimensionality);

    for( int j = 0; j < patterns.length; j++, current_pattern += patterns.rowStride) {
        densities.data[j] = kernel.densityFunction(
                current_pattern, patterns.dimensionality, kernelConstant);
    }

    /* Create return object */
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *single_pattern_symmetric(PyObject *args, KernelType kernelType) {
    /* Parse input data */
    PyObject* inPatterns = NULL;

    if (!PyArg_ParseTuple(args, "O", &inPatterns)) return NULL;

    Array pattern = pyObjectToArray(inPatterns, NPY_ARRAY_IN_ARRAY);

    /* Do computations */
    double density;

    SymmetricKernel kernel = selectSymmetricKernel(kernelType);

    double kernelConstant = kernel.factorFunction(pattern.dimensionality);
    density = kernel.densityFunction(pattern.data, pattern.dimensionality, kernelConstant);

    /* Create return object */
    PyObject *returnObject = Py_BuildValue("d", density);
    return returnObject;
}

PyObject *scaling_factor_symmetric(PyObject *args, KernelType kernelType) {
    /* Parse input data */
    double generalBandwidth;

    if (!PyArg_ParseTuple(args, "d", &generalBandwidth)) return NULL;

    /* Do computations */
    SymmetricKernel kernel = selectSymmetricKernel(kernelType);
    double scalingFactor = kernel.scalingFactorFunction(generalBandwidth);

    /* Create return object */
    PyObject *returnObject = Py_BuildValue("d", scalingFactor);
    return returnObject;
}

static char kernels_standardGaussian_docstring[] = "Evaluate the Standard Gaussian (zero vector mean and identity covariance matrix) for each row in the input matrix.";
static PyObject * standard_gaussian_multi_pattern(PyObject *self, PyObject *args){
    return multi_pattern_symmetric(args, STANDARD_GAUSSIAN);
}

static PyObject * standard_gaussian_single_pattern(PyObject *self, PyObject *args){
    return single_pattern_symmetric(args, STANDARD_GAUSSIAN);
}

static char kernels_standardGaussian_scaling_docstring[] = "Compute the scaling factor for the Standard Gaussian (zero vector mean and identity covariance matrix) kernel.";
static PyObject * standard_gaussian_scaling_factor(PyObject *self, PyObject *args){
    return scaling_factor_symmetric(args, STANDARD_GAUSSIAN);
}

static char kernels_gaussian_docstring[] = "Evaluate the Gaussian PDF for each row in the input matrix.";
static PyObject * gaussian_multi_pattern(PyObject *self, PyObject *args){
    /* Read input */
    PyObject* inPattern = NULL;
    PyObject* inMean = NULL;
    PyObject* inCovarianceMatrix = NULL;
    PyObject* outDensities = NULL;

    if (!PyArg_ParseTuple(args, "OOOO", &inPattern, &inMean, &inCovarianceMatrix, &outDensities)) return NULL;

    Array patterns = pyObjectToArray(inPattern, NPY_ARRAY_IN_ARRAY);
    Array mean = pyObjectToArray(inMean, NPY_ARRAY_IN_ARRAY);
    Array covarianceMatrix = pyObjectToArray(inCovarianceMatrix, NPY_ARRAY_IN_ARRAY);
    Array densities = pyObjectToArray(outDensities, NPY_ARRAY_OUT_ARRAY);

    gsl_vector_view mean_view = arrayGetGSLVectorView(&mean);
    gsl_matrix_view patterns_view = arrayGetGSLMatrixView(&patterns);

    /* Do computations */
    ASymmetricKernel kernel = selectASymmetricKernel(GAUSSIAN);
    gsl_matrix* kernelConstant = kernel.factorFunction(&covarianceMatrix);

    gsl_vector* current_pattern = gsl_vector_alloc((size_t) covarianceMatrix.dimensionality);

    for(int i = 0; i < patterns.length; i++){
        gsl_matrix_get_row(current_pattern, &patterns_view.matrix, i);
        densities.data[i] = kernel.densityFunction(current_pattern, &mean_view.vector, kernelConstant);
    }

    /* Free memory */
    gsl_vector_free(current_pattern);
    gsl_matrix_free(kernelConstant);

    /* Create return object */
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * gaussian_single_pattern(PyObject *self, PyObject *args){
    /* Read input */
    PyObject* inPattern = NULL;
    PyObject* inMean = NULL;
    PyObject* inCovarianceMatrix = NULL;

    if (!PyArg_ParseTuple(args, "OOO", &inPattern, &inMean, &inCovarianceMatrix)) return NULL;

    Array pattern = pyObjectToArray(inPattern, NPY_ARRAY_IN_ARRAY);
    Array mean = pyObjectToArray(inMean, NPY_ARRAY_IN_ARRAY);
    Array covarianceMatrix = pyObjectToArray(inCovarianceMatrix, NPY_ARRAY_IN_ARRAY);

    gsl_vector_view mean_view = arrayGetGSLVectorView(&mean);
    gsl_vector_view pattern_view = arrayGetGSLVectorView(&pattern);

    /* Do computations */
    ASymmetricKernel kernel = selectASymmetricKernel(GAUSSIAN);
    gsl_matrix* kernelConstant = kernel.factorFunction(&covarianceMatrix);
    double density = kernel.densityFunction(&pattern_view.vector, &mean_view.vector, kernelConstant);

    /* Free memory */
    gsl_matrix_free(kernelConstant);

    /* Create return object */
    PyObject *returnObject = Py_BuildValue("d", density);
    return returnObject;
}

static char kernels_gaussian_scaling_docstring[] = "Compute the scaling factor for the Gaussian kernel.";
static PyObject * gaussian_scaling_factor(PyObject *self, PyObject *args){
    /* Create temporary return object */
    Py_INCREF(Py_None);
    return Py_None;
}

static char kernels_epanechnikov_docstring[] = "Evaluate the Epanechnikov kernel for each row in the input matrix.";
static PyObject * epanechnikov_single_pattern(PyObject *self, PyObject *args){
    return single_pattern_symmetric(args, EPANECHNIKOV);
}

static PyObject * epanechnikov_multi_pattern(PyObject *self, PyObject *args){
    return multi_pattern_symmetric(args, EPANECHNIKOV);
}

static char kernels_epanechnikov_scaling_docstring[] = "Compute the scaling factor for the Epanechnikov kernel.";
static PyObject * epanechnikov_scaling_factor(PyObject *self, PyObject *args){
    return scaling_factor_symmetric(args, EPANECHNIKOV);
}

static char kernels_testKernel_docstring[] = "Evaluate the TestKernel for each row in the input matrix. This kernel returns the absolute value of the mean of the elements of the patterns.";
static PyObject * testKernel_single_pattern(PyObject *self, PyObject *args){
    return single_pattern_symmetric(args, TEST);
}

static PyObject * testKernel_multi_pattern(PyObject *self, PyObject *args){
    return multi_pattern_symmetric(args, TEST);
}

static char kernels_testKernel_scaling_docstring[] = "Compute the scaling factor for the Test kernel.";
static PyObject * testKernel_scaling_factor(PyObject *self, PyObject *args){
    return scaling_factor_symmetric(args, TEST);
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
        {"standard_gaussian_multi_pattern",     standard_gaussian_multi_pattern,    METH_VARARGS,   kernels_standardGaussian_docstring},
        {"standard_gaussian_single_pattern",    standard_gaussian_single_pattern,   METH_VARARGS,   kernels_standardGaussian_docstring},
        {"standard_gaussian_scaling_factor",    standard_gaussian_scaling_factor,   METH_VARARGS,   kernels_standardGaussian_scaling_docstring},

        {"gaussian_multi_pattern",              gaussian_multi_pattern,             METH_VARARGS,   kernels_gaussian_docstring},
        {"gaussian_single_pattern",             gaussian_single_pattern,            METH_VARARGS,   kernels_gaussian_docstring},
        {"gaussian_scaling_factor",             gaussian_multi_pattern,             METH_VARARGS,   kernels_gaussian_scaling_docstring},

        {"epanechnikov_single_pattern",         epanechnikov_single_pattern,        METH_VARARGS,   kernels_epanechnikov_docstring},
        {"epanechnikov_multi_pattern",          epanechnikov_multi_pattern,         METH_VARARGS,   kernels_epanechnikov_docstring},
        {"epanechnikov_scaling_factor",         epanechnikov_scaling_factor,        METH_VARARGS,   kernels_epanechnikov_scaling_docstring},

        {"test_kernel_single_pattern",          testKernel_single_pattern,          METH_VARARGS,   kernels_testKernel_docstring},
        {"test_kernel_multi_pattern",           testKernel_multi_pattern,           METH_VARARGS,   kernels_testKernel_docstring},
        {"test_kernel_scaling_factor",          testKernel_scaling_factor,          METH_VARARGS,   kernels_testKernel_scaling_docstring},
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