#include "utilsModule.h"

static char utils_distanceMatrix_docstring[] = "Compute the distance matrix for the input patterns with squared Euclidean distance as a metric.";
static PyObject * distance_matrix(PyObject *self, PyObject *args){

    /* Handle input */
    PyObject* inPatterns = NULL;
    PyObject* outDistanceMatrix = NULL;

    if (!PyArg_ParseTuple(args, "OO", &inPatterns, &outDistanceMatrix)) return NULL;

    Array patterns = pyObjectToArray(inPatterns, NPY_ARRAY_IN_ARRAY);
    Array distanceMatrix = pyObjectToArray(outDistanceMatrix, NPY_ARRAY_OUT_ARRAY);

    /* Do stuff */
    computeDistanceMatrix(&patterns, &distanceMatrix);

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
        {"distance_matrix",     distance_matrix,    METH_VARARGS,   utils_distanceMatrix_docstring},
        {"knn",                 knn,                METH_VARARGS,   utils_knn_docstring},
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