#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector_double.h>
#include "kernelsModule.h"

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

static char kernels_standardGaussian_docstring[] = "Evaluate the Standard Gaussian (zero vector mean and identity covariance matrix) for each row in the input matrix.";
static PyObject * standard_gaussian_multi_pattern(PyObject *self, PyObject *args){
    return multi_pattern_symmetric(args, STANDARD_GAUSSIAN);
}

static PyObject * standard_gaussian_single_pattern(PyObject *self, PyObject *args){
    return single_pattern_symmetric(args, STANDARD_GAUSSIAN);
}

static char kernels_gaussian_docstring[] = "Evaluate the Gaussian PDF for each row in the input matrix.";
static PyObject * gaussian_multi_pattern(PyObject *self, PyObject *args){
    return multi_pattern(args, GAUSSIAN);
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
    /* Do computations */
    double density;

    ASymmetricKernel kernel = selectASymmetricKernel(GAUSSIAN);
    gsl_matrix* kernelConstant = kernel.factorFunction(&covarianceMatrix);
    density = kernel.densityFunction(pattern.data, &mean_view.vector, kernelConstant);

    /* Create return object */
    PyObject *returnObject = Py_BuildValue("d", density);
    return returnObject;
}

static char kernels_epanechnikov_docstring[] = "Evaluate the Epanechnikov kernel for each row in the input matrix.";
static PyObject * epanechnikov_single_pattern(PyObject *self, PyObject *args){
    return single_pattern_symmetric(args, EPANECHNIKOV);
}

static PyObject * epanechnikov_multi_pattern(PyObject *self, PyObject *args){
    return multi_pattern_symmetric(args, EPANECHNIKOV);
}

static char kernels_testKernel_docstring[] = "Evaluate the TestKernel for each row in the input matrix. This kernel returns the absolute value of the mean of the elements of the patterns.";
static PyObject * testKernel_single_pattern(PyObject *self, PyObject *args){
    return single_pattern_symmetric(args, TEST);
}

static PyObject * testKernel_multi_pattern(PyObject *self, PyObject *args){
    return multi_pattern_symmetric(args, TEST);
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
        {"gaussian_multi_pattern",              gaussian_multi_pattern,             METH_VARARGS,   kernels_gaussian_docstring},
        {"gaussian_single_pattern",             gaussian_single_pattern,            METH_VARARGS,   kernels_gaussian_docstring},
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