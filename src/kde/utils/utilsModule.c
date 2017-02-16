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
    PyObject *returnObject = Py_BuildValue("d", 42.0);
    return returnObject;
}

Array pyObjectToArray(PyObject *pythonObject, int requirements){
    PyArrayObject* arrayObject = NULL;
    arrayObject = (PyArrayObject *)PyArray_FROM_OTF(pythonObject, NPY_DOUBLE, requirements);
    if (arrayObject == NULL){
        fprintf(stderr, "Error converting PyObject to PyArrayObject\n");
        exit(-1);
    }
    Array array = buildArrayFromPyArray(arrayObject);
    Py_XDECREF(arrayObject);
    return array;
}

static PyMethodDef method_table[] = {
        {"distance_matrix",     distance_matrix,    METH_VARARGS,   utils_distanceMatrix_docstring},
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