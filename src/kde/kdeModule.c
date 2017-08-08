#include "kdeModule.h"

static char kde_parzen_docstring[] = "Estimate densities with Parzen.";
static PyObject * kdeParzen(PyObject *self, PyObject *args){
    PyObject* inPatterns = NULL;
    PyObject* inDataPoints = NULL;
    PyObject* outDensities = NULL;
    PyObject* outNumUsedPatterns = NULL;

    double inWindowWidth;
    KernelType kernelType;

    if (!PyArg_ParseTuple(args, "OOdiOO",
                          &inPatterns,
                          &inDataPoints,
                          &inWindowWidth,
                          &kernelType,
                          &outDensities,
                          &outNumUsedPatterns)) return NULL;

    gsl_matrix_view xs = pyObjectToGSLMatrixView(inPatterns, NPY_ARRAY_IN_ARRAY);
    gsl_matrix_view xis = pyObjectToGSLMatrixView(inDataPoints, NPY_ARRAY_IN_ARRAY);
    gsl_vector_view densities = pyObjectToGSLVectorView(outDensities, NPY_ARRAY_OUT_ARRAY);
    gsl_vector_view numUsedPatterns = pyObjectToGSLVectorView(outNumUsedPatterns, NPY_ARRAY_OUT_ARRAY);

    parzen(&xs.matrix, &xis.matrix, inWindowWidth, kernelType, 
      &densities.vector, &numUsedPatterns.vector
    );

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
    PyObject* outNumUsedPatterns = NULL;


    double globalBandwidth;
    KernelType inKernelType;

    if (!PyArg_ParseTuple(args, "OOdOiOO",
                          &inPatterns,
                          &inDataPoints,
                          &globalBandwidth,
                          &inLocalBandwidths,
                          &inKernelType,
                          &outDensities,
                          &outNumUsedPatterns)) return NULL;

    gsl_matrix_view xs = pyObjectToGSLMatrixView(inPatterns, NPY_ARRAY_IN_ARRAY);
    gsl_matrix_view xis = pyObjectToGSLMatrixView(inDataPoints, NPY_ARRAY_IN_ARRAY);
    gsl_vector_view localBandwidths = pyObjectToGSLVectorView(inLocalBandwidths, NPY_ARRAY_IN_ARRAY);
    gsl_vector_view densities = pyObjectToGSLVectorView(outDensities, NPY_ARRAY_OUT_ARRAY);
    gsl_vector_view numUsedPatterns = pyObjectToGSLVectorView(outNumUsedPatterns, NPY_ARRAY_OUT_ARRAY);

    mbe(&xs.matrix, &xis.matrix,
        globalBandwidth, &localBandwidths.vector,
        inKernelType,
        &densities.vector, &numUsedPatterns.vector);

    /* Create return object */
    Py_INCREF(Py_None);
    return Py_None;
}

static char kde_sambe_docstring[] = "Perform the final estimation step of the Shape Adaptive Modified breiman estimator.";
static PyObject *kde_shape_adaptive_mbe(PyObject *self, PyObject *args){

    /* Parse Arguments */
    PyObject* inXs = NULL;
    PyObject* inXis = NULL;
    PyObject* inLocalBandwidths = NULL;
    PyObject* outDensities = NULL;
    PyObject* outNumUsedPatterns = NULL;
    PyObject* outEigenValues = NULL;
    PyObject* outEigenVectors = NULL;
    PyObject* outScalingFactors = NULL;
    KernelType kernelType;
    int k;
    double globalBandwidth;


    if (!PyArg_ParseTuple(args, "OOiidOOOOOO",
                          &inXs,
                          &inXis,
                          &kernelType,
                          &k,
                          &globalBandwidth,
                          &inLocalBandwidths,
                          &outDensities,
                          &outNumUsedPatterns,
                          &outEigenValues,
                          &outEigenVectors,
                          &outScalingFactors)) return NULL;

    gsl_matrix_view xs = pyObjectToGSLMatrixView(inXs, NPY_ARRAY_IN_ARRAY);
    gsl_matrix_view xis = pyObjectToGSLMatrixView(inXis, NPY_ARRAY_IN_ARRAY);
    gsl_vector_view localBandwidths = pyObjectToGSLVectorView(inLocalBandwidths, NPY_ARRAY_IN_ARRAY);
    gsl_vector_view densities = pyObjectToGSLVectorView(outDensities, NPY_ARRAY_OUT_ARRAY);
    gsl_vector_view numUsedPatterns = pyObjectToGSLVectorView(outNumUsedPatterns, NPY_ARRAY_OUT_ARRAY);
    gsl_matrix_view eigenValues = pyObjectToGSLMatrixView(outEigenValues, NPY_ARRAY_OUT_ARRAY);
    gsl_matrix_view eigenVectors = pyObjectToGSLMatrixView(outEigenVectors, NPY_ARRAY_OUT_ARRAY);
    gsl_vector_view scalingFactors = pyObjectToGSLVectorView(outScalingFactors, NPY_ARRAY_OUT_ARRAY);

    gsl_vector_set(&scalingFactors.vector, 0, 44);

    /* Do computations */
    sambe(
      &xs.matrix, &xis.matrix, 
      &localBandwidths.vector, globalBandwidth,
      kernelType, k,
      &densities.vector, &numUsedPatterns.vector,
      &eigenValues.matrix, &eigenVectors.matrix, &scalingFactors.vector
    );

    /* Create return object */
    Py_INCREF(Py_None);
    return Py_None;
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

PyMODINIT_FUNC init_kde(void) {
    (void)Py_InitModule("_kde", method_table);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
    import_array();
#pragma GCC diagnostic pop   
}