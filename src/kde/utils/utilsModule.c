#include <gsl/gsl_matrix.h>
#include "utilsModule.h"

static char utils_distanceMatrix_docstring[] = "Compute the distance matrix for the input patterns with squared Euclidean distance as a metric.";
static PyObject * distance_matrix(PyObject *self, PyObject *args){

    /* Handle input */
    PyObject* inPatterns = NULL;
    PyObject* outDistanceMatrix = NULL;

    if (!PyArg_ParseTuple(args, "OO", &inPatterns, &outDistanceMatrix)) return NULL;


    gsl_matrix_view patterns = pyObjectToGSLMatrixView(inPatterns, NPY_ARRAY_IN_ARRAY);
    gsl_matrix_view distanceMatrix = pyObjectToGSLMatrixView(outDistanceMatrix, NPY_ARRAY_OUT_ARRAY);

    /* Do stuff */
    computeDistanceMatrix(&patterns.matrix, &distanceMatrix.matrix);

    /* Create return object */
    Py_INCREF(Py_None);
    return Py_None;
}

static char utils_knn_docstring[] = "Compute the K nearest neighbours of some pattern, based on the provided distance matrix.";
static PyObject * knn(PyObject *self, PyObject *args){

    /* Handle input */
    PyObject* inPatterns = NULL;
    PyObject* inDistanceMatrix = NULL;
    PyObject* outNearestNeighbours = NULL;

    int k;
    int patternIdx;

    if (!PyArg_ParseTuple(args, "iiOOO",
                          &k, &patternIdx, &inPatterns, &inDistanceMatrix, &outNearestNeighbours)) return NULL;

    Array patterns = pyObjectToArray(inPatterns, NPY_ARRAY_IN_ARRAY);
    Array distanceMatrix = pyObjectToArray(inDistanceMatrix, NPY_ARRAY_IN_ARRAY);
    Array nearestNeighbours = pyObjectToArray(outNearestNeighbours, NPY_ARRAY_OUT_ARRAY);


    /* Do stuff */
    compute_k_nearest_neighbours(k, patternIdx, &patterns, &distanceMatrix, &nearestNeighbours);

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

    Array patterns = pyObjectToArray(inPatterns, NPY_ARRAY_IN_ARRAY);
    Array covarianceMatrix = pyObjectToArray(outCovarianceMatrix, NPY_ARRAY_OUT_ARRAY);

    /* Do stuff */
    computeCovarianceMatrix(&patterns, &covarianceMatrix);

    /* Create return object */
    Py_INCREF(Py_None);
    return Py_None;
}

static char utils_eigenValues_docstring[] = "Compute the eigen values of the input matrix.";
static PyObject * eigenValues(PyObject *self, PyObject *args){

    /* Handle input */
    PyObject* inMatrix = NULL;
    PyObject* outEigenValues = NULL;

    if (!PyArg_ParseTuple(args, "OO",
                          &inMatrix, &outEigenValues)) return NULL;

    Array matrix = pyObjectToArray(inMatrix, NPY_ARRAY_IN_ARRAY);
    Array eigenValues = pyObjectToArray(outEigenValues, NPY_ARRAY_OUT_ARRAY);

    /* Do stuff */
    computeEigenValues(&matrix, &eigenValues);

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
        {"distance_matrix",     distance_matrix,    METH_VARARGS,   utils_distanceMatrix_docstring},
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