#include "kernelsModule.h"


PyObject *multi_pattern_symmetric(PyObject *args, KernelType kernelType) {    
    /* Parse input data */
    PyObject* inPatterns = NULL;
    PyObject* outDensities = NULL;

    if (!PyArg_ParseTuple(args, "OO", &inPatterns, &outDensities)) return NULL;

    gsl_matrix_view patterns = pyObjectToGSLMatrixView(inPatterns, NPY_ARRAY_IN_ARRAY);
    gsl_vector_view densities = pyObjectToGSLVectorView(outDensities, NPY_ARRAY_OUT_ARRAY);

    int numThreads = 1;
    #pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }

    /* Do computations */
    SymmetricKernel kernel = selectSymmetricKernel(kernelType);
    kernel.prepare(patterns.matrix.size2, numThreads);

    #pragma omp parallel shared(patterns, densities)
    {
        gsl_vector_view current_pattern;
        double density;
        int pid = omp_get_thread_num();
        
        #pragma omp for
        for(size_t j = 0; j < patterns.matrix.size1; j++) {
            current_pattern = gsl_matrix_row(&patterns.matrix, j);
            density = kernel.density(&current_pattern.vector, pid);
            gsl_vector_set(&densities.vector, j, density);
        }
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

    gsl_matrix_view patternMatrix = pyObjectToGSLMatrixView(inPattern, NPY_ARRAY_IN_ARRAY);

    gsl_vector_view pattern = gsl_matrix_row(&patternMatrix.matrix, 0);

    /* Do computations */
    double density;

    SymmetricKernel kernel = selectSymmetricKernel(kernelType);

    kernel.prepare(pattern.vector.size, 1);
    density = kernel.density(&pattern.vector, 0);
    kernel.free();

    /* Create return object */
    PyObject *returnObject = Py_BuildValue("d", density);
    return returnObject;
}

PyObject *single_pattern_shape_adaptive(PyObject *args, KernelType kernelType){
    /* Read input */
    PyObject* inPattern = NULL;
    PyObject* inGlobalBandwidthMatrix = NULL;

    if (!PyArg_ParseTuple(args, "OO", &inPattern, &inGlobalBandwidthMatrix)) return NULL;

    gsl_matrix_view patternMatrix = pyObjectToGSLMatrixView(inPattern, NPY_ARRAY_IN_ARRAY);
    gsl_vector_view pattern = gsl_matrix_row(&patternMatrix.matrix, 0);

    gsl_matrix_view globalBandwidthMatrix = pyObjectToGSLMatrixView(inGlobalBandwidthMatrix, NPY_ARRAY_IN_ARRAY);

    ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(kernelType);

    int numThreads = 1;
    int pid = 0;

    kernel.allocate(pattern.vector.size, numThreads);
    kernel.computeConstants(&globalBandwidthMatrix.matrix, pid);

    double density = kernel.density(&pattern.vector, pid);

    /* Free memory */
    kernel.free();

    /* Create return object */
    PyObject *returnObject = Py_BuildValue("d", density);
    return returnObject;
}

PyObject *multiple_patterns_shape_adaptive(PyObject *args, KernelType kernelType) {
    /* Read input */
    PyObject* inPatterns = NULL;
    PyObject* inGlobalBandwidthMatrix = NULL;
    PyObject* outDensities = NULL;

    if (!PyArg_ParseTuple(args, "OOO",
                          &inPatterns,
                          &inGlobalBandwidthMatrix,
                          &outDensities)) return NULL;

    gsl_matrix_view patterns = pyObjectToGSLMatrixView(inPatterns, NPY_ARRAY_IN_ARRAY);
    gsl_matrix* globalBandwidthMatrix = pyObjectToGSLMatrix(inGlobalBandwidthMatrix, NPY_ARRAY_IN_ARRAY);
    gsl_vector_view densities = pyObjectToGSLVectorView(outDensities, NPY_ARRAY_OUT_ARRAY);

    ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(kernelType);

    int numThreads = 1;
    #pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }

    kernel.allocate(patterns.matrix.size2, numThreads);

    #pragma omp parallel
    {
        int pid = omp_get_thread_num();

        double density;
        gsl_vector_view pattern;

        kernel.computeConstants(globalBandwidthMatrix, pid);

        #pragma omp for
        for(size_t j = 0; j < patterns.matrix.size1; j++) {
            pattern = gsl_matrix_row(&patterns.matrix, j);
            density = kernel.density(&pattern.vector, pid);
            gsl_vector_set(&densities.vector, j, density);
        }        
    }

    kernel.free();

    /* Create return object */
    Py_INCREF(Py_None);
    return Py_None;
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
    return single_pattern_shape_adaptive(args, SHAPE_ADAPTIVE_GAUSSIAN);
}

static PyObject * sa_gaussian_multi_pattern(PyObject *self, PyObject *args){
    return multiple_patterns_shape_adaptive(args, SHAPE_ADAPTIVE_GAUSSIAN);
}

static char kernels_sa_epanechnikov_docstring[] = "Evaluate the shape adaptive Epanechnikov kernel for each row in the input matrix.";
static PyObject * sa_epanechnikov_single_pattern(PyObject *self, PyObject *args){
    return single_pattern_shape_adaptive(args, SHAPE_ADAPTIVE_EPANECHNIKOV);
}

static PyObject * sa_epanechnikov_multi_pattern(PyObject *self, PyObject *args){
    return multiple_patterns_shape_adaptive(args, SHAPE_ADAPTIVE_EPANECHNIKOV);
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
    double localBandwidth, generalBandwidth;
    PyObject* inCovarianceMatrix = NULL;

    if (!PyArg_ParseTuple(args, "ddO", &localBandwidth, &generalBandwidth, &inCovarianceMatrix)) return NULL;

    gsl_matrix_view covarianceMatrix = pyObjectToGSLMatrixView(inCovarianceMatrix, NPY_ARRAY_IN_ARRAY);

    /* Do computations */
    double scalingFactor = computeScalingFactor(localBandwidth, generalBandwidth, &covarianceMatrix.matrix);

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

    size_t length = (size_t) PyArray_DIM(arrayObject, 0);

    Py_XDECREF(arrayObject);
    return gsl_vector_view_array(data, length);
}

static PyMethodDef method_table[] = {
        {"standard_gaussian_multi_pattern",     standard_gaussian_multi_pattern,    METH_VARARGS,   kernels_standardGaussian_docstring},
        {"standard_gaussian_single_pattern",    standard_gaussian_single_pattern,   METH_VARARGS,   kernels_standardGaussian_docstring},

        {"epanechnikov_single_pattern",         epanechnikov_single_pattern,        METH_VARARGS,   kernels_epanechnikov_docstring},
        {"epanechnikov_multi_pattern",          epanechnikov_multi_pattern,         METH_VARARGS,   kernels_epanechnikov_docstring},

        {"test_kernel_single_pattern",          testKernel_single_pattern,          METH_VARARGS,   kernels_testKernel_docstring},
        {"test_kernel_multi_pattern",           testKernel_multi_pattern,           METH_VARARGS,   kernels_testKernel_docstring},

        {"sa_gaussian_single_pattern",          sa_gaussian_single_pattern,         METH_VARARGS,   kernels_sa_gaussian_docstring},
        {"sa_gaussian_multi_pattern",           sa_gaussian_multi_pattern,          METH_VARARGS,   kernels_sa_gaussian_docstring},

        {"sa_epanechnikov_single_pattern",      sa_epanechnikov_single_pattern,     METH_VARARGS,   kernels_sa_epanechnikov_docstring},
        {"sa_epanechnikov_multi_pattern",       sa_epanechnikov_multi_pattern,      METH_VARARGS,   kernels_sa_epanechnikov_docstring},

        {"scaling_factor",                      scaling_factor,                     METH_VARARGS,   kernels_scalingFactor_docstring},
        /* Sentinel */
        {NULL,                              NULL,                                   0,              NULL}
};

PyMODINIT_FUNC init_kernels(void) {
    (void)Py_InitModule("_kernels", method_table);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
    import_array();
#pragma GCC diagnostic pop   
}