#include "utilsModule.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector_double.h>

static char utils_knn_docstring[] = "Compute the K nearest neighbours of some pattern, based on the provided distance matrix.";
static PyObject * knn(PyObject *self, PyObject *args){

    /* Handle input */
    PyObject* inPatterns = NULL;
    PyObject* outNearestNeighbours = NULL;

    int k;
    int patternIdx;

    if (!PyArg_ParseTuple(args, "iiOO",
                          &k, &patternIdx, &inPatterns, &outNearestNeighbours)) return NULL;

    gsl_matrix_view patterns = pyObjectToGSLMatrixView(inPatterns, NPY_ARRAY_IN_ARRAY);
    gsl_matrix_view nearestNeighbours = pyObjectToGSLMatrixView(outNearestNeighbours, NPY_ARRAY_OUT_ARRAY);

    /* Do stuff */
    nn_prepare(&patterns.matrix);
    computeKNearestNeighboursOld(k, patternIdx, &patterns.matrix,
                                 &nearestNeighbours.matrix);
    nn_free();

    /* Create return object */
    Py_INCREF(Py_None);
    return Py_None;
}

static char utils_covarianceMatrix_docstring[] = "Compute the covariance matrix of the data.";
static PyObject * covariance_matrix(PyObject *self, PyObject *args){

    /* Handle input */
    PyObject* inPatterns = NULL;
    PyObject* outCovarianceMatrix = NULL;

    if (!PyArg_ParseTuple(args, "OO",
                          &inPatterns, &outCovarianceMatrix)) return NULL;

    gsl_matrix_view patterns = pyObjectToGSLMatrixView(inPatterns, NPY_ARRAY_IN_ARRAY);
    gsl_matrix_view covarianceMatrix = pyObjectToGSLMatrixView(outCovarianceMatrix, NPY_ARRAY_OUT_ARRAY);

    /* Do stuff */
    computeCovarianceMatrix(&patterns.matrix, &covarianceMatrix.matrix);

    /* Create return object */
    Py_INCREF(Py_None);
    return Py_None;
}

static char utils_eigenValues_docstring[] = "Compute the eigen values of the input matrix.";

static gsl_vector_view pyObjectToGSLVector(PyObject *pObject, int i);

static PyObject * eigenValues(PyObject *self, PyObject *args){

    /* Handle input */
    PyObject* inMatrix = NULL;
    PyObject* outEigenValues = NULL;

    if (!PyArg_ParseTuple(args, "OO",
                          &inMatrix, &outEigenValues)) return NULL;

    gsl_matrix_view matrix = pyObjectToGSLMatrixView(inMatrix, NPY_ARRAY_IN_ARRAY);
    gsl_vector_view eigenvalues = pyObjectToGSLVector(outEigenValues, NPY_ARRAY_OUT_ARRAY);

    /* Do stuff */
    computeEigenValues(&matrix.matrix, &eigenvalues.vector);

    /* Create return object */
    Py_INCREF(Py_None);
    return Py_None;
}

static char utils_geometricmean_docstring[] = "Compute the geometric mean of a set of values.";
static PyObject * geometric_mean(PyObject *self, PyObject *args){

    /* Handle input */
    PyObject* inValues = NULL;

    if (!PyArg_ParseTuple(args, "O", &inValues)) return NULL;

    Array values = pyObjectToArray(inValues, NPY_ARRAY_IN_ARRAY);

    /* Do stuff */
    double geometricMean = computeGeometricMean(values.data, (size_t) values.length);

    /* Create return object */
    PyObject *returnObject = Py_BuildValue("d", geometricMean);
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

gsl_vector_view pyObjectToGSLVector(PyObject *pythonObject, int requirements) {
    PyArrayObject* arrayObject = NULL;
    arrayObject = (PyArrayObject *)PyArray_FROM_OTF(pythonObject, NPY_DOUBLE, requirements);
    if (arrayObject == NULL){
        fprintf(stderr, "Error converting PyObject to PyArrayObject\n");
        exit(-1);
    }
    double* data = (double *)PyArray_DATA(arrayObject);
    size_t size = PyArray_DIM(arrayObject, 0);

    Py_XDECREF(arrayObject);
    return gsl_vector_view_array(data, size);
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

static PyMethodDef method_table[] = {
        {"knn",                 knn,                METH_VARARGS,   utils_knn_docstring},
        {"covariance_matrix",   covariance_matrix,  METH_VARARGS,   utils_covarianceMatrix_docstring},
        {"eigen_values",        eigenValues,        METH_VARARGS,   utils_eigenValues_docstring},
        {"geometric_mean",      geometric_mean,     METH_VARARGS,   utils_geometricmean_docstring},
        /* Sentinel */
        {NULL,                  NULL,               0,              NULL}
};

static struct PyModuleDef utilsModule = {
        PyModuleDef_HEAD_INIT, "_utils",
        "C implementation of some utility functions.",
        -1, method_table
};

PyMODINIT_FUNC PyInit__utils(void) {
    PyObject *module = PyModule_Create(&utilsModule);

    if(!module) return NULL;
    import_array();

    return module;
}