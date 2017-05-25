#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector_double.h>
#include "kernelsModule.h"
#include "kernels.h"

PyObject *multi_pattern_symmetric(PyObject *args, KernelType kernelType) {
    /* Parse input data */
    PyObject* inPatterns = NULL;
    PyObject* outDensities = NULL;

    if (!PyArg_ParseTuple(args, "OO", &inPatterns, &outDensities)) return NULL;

    gsl_matrix_view patterns = pyObjectToGSLMatrixView(inPatterns, NPY_ARRAY_IN_ARRAY);
    gsl_vector_view densities = pyObjectToGSLVectorView(outDensities, NPY_ARRAY_OUT_ARRAY);

    /* Do computations */
    SymmetricKernel kernel = selectSymmetricKernel(kernelType);

    gsl_vector_view current_pattern;
    double density;
    kernel.prepare(patterns.matrix.size2);

    for( size_t j = 0; j < patterns.matrix.size1; j++) {
        current_pattern = gsl_matrix_row(&patterns.matrix, j);
        density = kernel.density(&current_pattern.vector);
        gsl_vector_set(&densities.vector, j, density);
    }
    kernel.free();

    /* Create return object */
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *single_pattern_symmetric(PyObject *args, KernelType kernelType) {
    /* Parse input data */
    PyObject* inPattern = NULL;

    if (!PyArg_ParseTuple(args, "O", &inPattern)) return NULL;

    gsl_vector_view pattern = pyObjectToGSLVectorView(inPattern, NPY_ARRAY_IN_ARRAY);

    /* Do computations */
    double density;

    SymmetricKernel kernel = selectSymmetricKernel(kernelType);

    kernel.prepare(pattern.vector.size);
    density = kernel.density(&pattern.vector);
    kernel.free();

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

static char kernels_sa_gaussian_docstring[] = "Evaluate the shape adaptive gaussian kernel for each row in the input matrix.";
static PyObject * sa_gaussian_single_pattern(PyObject *self, PyObject *args){
    /* Read input */
    PyObject* inPattern = NULL;
    PyObject* inGlobalBandwidthMatrix = NULL;

    double localBandwidth;

    if (!PyArg_ParseTuple(args, "OdO", &inPattern, &localBandwidth, &inGlobalBandwidthMatrix)) return NULL;

    Array pattern = pyObjectToArray(inPattern, NPY_ARRAY_IN_ARRAY);
    gsl_matrix* globalBandwidthMatrix = pyObjectToGSLMatrix(inGlobalBandwidthMatrix, NPY_ARRAY_IN_ARRAY);


    /* Compute constants */
    size_t dimension = (size_t) pattern.dimensionality;
    ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(SHAPE_ADAPTIVE_GAUSSIAN);
    gsl_matrix* globalInverse = gsl_matrix_alloc(dimension, dimension);
    double globalScalingFactor, pdfConstant;

    kernel.factorFunction(globalBandwidthMatrix, globalInverse, &globalScalingFactor, &pdfConstant);

    /* Allocate Memory for the kernel evaluation */
    gsl_vector* scaledPatternMemory = gsl_vector_calloc(dimension);

    kernel.allocate(dimension);
    kernel.computeConstants(globalBandwidthMatrix);

    /* Do computations */
    gsl_vector_view pattern_view = arrayGetGSLVectorView(&pattern);
    double density = kernel.density(&pattern_view.vector, localBandwidth);

    /* Free memory */
    kernel.free();
    gsl_matrix_free(globalInverse);
    gsl_vector_free(scaledPatternMemory);
    gsl_matrix_free(globalBandwidthMatrix);

    /* Create return object */
    PyObject *returnObject = Py_BuildValue("d", density);
    return returnObject;
}

static PyObject * sa_gaussian_multi_pattern(PyObject *self, PyObject *args){
    /* Read input */
    PyObject* inPatterns = NULL;
    PyObject* inLocalBandwidths = NULL;
    PyObject* inGlobalBandwidthMatrix = NULL;
    PyObject* outDensities = NULL;

    if (!PyArg_ParseTuple(args, "OOOO",
                          &inPatterns,
                          &inLocalBandwidths,
                          &inGlobalBandwidthMatrix,
                          &outDensities)) return NULL;

    Array patterns = pyObjectToArray(inPatterns, NPY_ARRAY_IN_ARRAY);
    gsl_matrix* globalBandwidthMatrix = pyObjectToGSLMatrix(inGlobalBandwidthMatrix, NPY_ARRAY_IN_ARRAY);
    Array localBandwidths = pyObjectToArray(inLocalBandwidths, NPY_ARRAY_IN_ARRAY);
    Array densities = pyObjectToArray(outDensities, NPY_ARRAY_OUT_ARRAY);

    ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(SHAPE_ADAPTIVE_GAUSSIAN);

    /* Compute constants */
    size_t dimension = (size_t) patterns.dimensionality;
    gsl_matrix* globalInverse = gsl_matrix_alloc(dimension, dimension);
    double globalScalingFactor, pdfConstant;
    kernel.factorFunction(globalBandwidthMatrix, globalInverse, &globalScalingFactor, &pdfConstant);

    /* Allocate memory for the kernel evaluatation */
    gsl_vector* scaledPatternMemory = gsl_vector_alloc(dimension);

    /* Do computations */
    double* pattern = patterns.data;
    double localBandwidth;

    gsl_vector_view pattern_view;

    kernel.allocate(dimension);
    kernel.computeConstants(globalBandwidthMatrix);

    for( int j = 0; j < patterns.length; j++, pattern += patterns.rowStride) {
        pattern_view = gsl_vector_view_array(pattern, (size_t) patterns.dimensionality);

        gsl_vector_set_zero(scaledPatternMemory);

        localBandwidth = localBandwidths.data[j];

        densities.data[j] = kernel.density(&pattern_view.vector, localBandwidth);
    }

    /* Free memory */
    kernel.free();
    gsl_matrix_free(globalInverse);
    gsl_matrix_free(globalBandwidthMatrix);
    gsl_vector_free(scaledPatternMemory);

    /* Create return object */
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

static char kernels_testKernel_docstring[] = "Evaluate the TestKernel for each row in the input matrix. This kernel returns the absolute value of the mean of the elements of the patterns.";
static PyObject * testKernel_single_pattern(PyObject *self, PyObject *args){
    return single_pattern_symmetric(args, TEST);
}

static PyObject * testKernel_multi_pattern(PyObject *self, PyObject *args){
    return multi_pattern_symmetric(args, TEST);
}

static char kernels_scalingFactor_docstring[] = "Compute the scaling factor for the asymmetric kernel.";
static PyObject* scaling_factor(PyObject* self, PyObject *args){
    /* Read input */
    double generalBandwidth;
    PyObject* inCovarianceMatrix = NULL;

    if (!PyArg_ParseTuple(args, "dO", &generalBandwidth, &inCovarianceMatrix)) return NULL;

    gsl_matrix_view covarianceMatrix = pyObjectToGSLMatrixView(inCovarianceMatrix, NPY_ARRAY_IN_ARRAY);

    /* Do computations */
    double scalingFactor = computeScalingFactor(generalBandwidth, &covarianceMatrix.matrix);

    /* Free memory */

    /* Create return object */
    PyObject *returnObject = Py_BuildValue("d", scalingFactor);
    return returnObject;
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

gsl_matrix_view pyObjectToGSLMatrixView(PyObject *pythonObject, int requirements) {
    PyArrayObject* arrayObject = NULL;
    arrayObject = (PyArrayObject *)PyArray_FROM_OTF(pythonObject, NPY_DOUBLE, requirements);
    if (arrayObject == NULL){
        fprintf(stderr, "Error converting PyObject to PyArrayObject\n");
        exit(-1);
    }
    double* data = (double *)PyArray_DATA(arrayObject);
    size_t num_rows = PyArray_DIM(arrayObject, 0);
    size_t num_cols = PyArray_DIM(arrayObject, 1);

    Py_XDECREF(arrayObject);
    return gsl_matrix_view_array(data, num_rows, num_cols);
}

gsl_matrix *pyObjectToGSLMatrix(PyObject *pythonObject, int requirements) {
    gsl_matrix_view view = pyObjectToGSLMatrixView(pythonObject, requirements);

    gsl_matrix* matrix = gsl_matrix_alloc(view.matrix.size1, view.matrix.size2);
    gsl_matrix_memcpy(matrix, &view.matrix);
    return matrix;
}

gsl_vector_view pyObjectToGSLVectorView(PyObject *pythonObject, int requirements) {
    PyArrayObject* arrayObject = NULL;
    arrayObject = (PyArrayObject *)PyArray_FROM_OTF(pythonObject, NPY_DOUBLE, requirements);
    if (arrayObject == NULL){
        fprintf(stderr, "Error converting PyObject to PyArrayObject\n");
        exit(-1);
    }
    double* data = (double *)PyArray_DATA(arrayObject);

    size_t length = (size_t) PyArray_DIM(arrayObject, 1);

    Py_XDECREF(arrayObject);
    return gsl_vector_view_array(data, length);
}

static PyMethodDef method_table[] = {
        {"standard_gaussian_multi_pattern",     standard_gaussian_multi_pattern,    METH_VARARGS,   kernels_standardGaussian_docstring},
        {"standard_gaussian_single_pattern",    standard_gaussian_single_pattern,   METH_VARARGS,   kernels_standardGaussian_docstring},

        {"epanechnikov_single_pattern",         epanechnikov_single_pattern,        METH_VARARGS,   kernels_epanechnikov_docstring},
        {"epanechnikov_multi_pattern",          epanechnikov_multi_pattern,         METH_VARARGS,   kernels_epanechnikov_docstring},

        {"test_kernel_single_pattern",          testKernel_single_pattern,          METH_VARARGS,   kernels_testKernel_docstring},
        {"test_kernel_multi_pattern",           testKernel_multi_pattern,  /**/     METH_VARARGS,   kernels_testKernel_docstring},

        {"sa_gaussian_single_pattern",          sa_gaussian_single_pattern,         METH_VARARGS,   kernels_sa_gaussian_docstring},
        {"sa_gaussian_multi_pattern",           sa_gaussian_multi_pattern,          METH_VARARGS,   kernels_sa_gaussian_docstring},

        {"scaling_factor",                      scaling_factor,                     METH_VARARGS,   kernels_scalingFactor_docstring},
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