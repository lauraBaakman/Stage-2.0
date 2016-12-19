//
// Created by Laura Baakman on 19/12/2016.
//

#include "kernels.h"

static char kernels_standardGaussian_docstring[] = "Evaluate the Standard Gaussian (zero vector mean and identity covariance matrix) for each row in the input matrix.";
static PyObject * kernels_standard_gaussian(PyObject *self, PyObject *args){
    PyObject* inPatterns = NULL;
    PyObject* outDensities = NULL;

    PyArrayObject* patterns = NULL;
    PyArrayObject* densities = NULL;

    if (!PyArg_ParseTuple(args, "OO", &inPatterns, &outDensities))
        goto fail;

    patterns = (PyArrayObject *)PyArray_FROM_OTF(inPatterns, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (patterns == NULL) goto fail;

    densities = (PyArrayObject *)PyArray_FROM_OTF(outDensities, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);
    if (densities == NULL) goto fail;

    double* densities_data = (double *)PyArray_DATA(densities);

    int num_patterns = (int)PyArray_DIM(patterns, 0);
    int dim_patterns = (int)PyArray_DIM(patterns, 1);

    int pattern_stride = (int)PyArray_STRIDE (patterns, 0) / (int)PyArray_ITEMSIZE(patterns);

    double* current_pattern = (double*)PyArray_DATA(patterns);

    for(int j = 0; j < num_patterns; j++)
    {
        densities_data[j] = standard_gaussian(current_pattern, dim_patterns);
        current_pattern += pattern_stride;
    }

//    /* Call the function that does the computation */
//    double result = term;
//
//    int dim_0 = PyArray_DIM(vector, 0);
//    printf("dim_0: %d\n", dim_0);
//
//    int dim_1 = PyArray_DIM(vector, 1);
//    printf("dim_1: %d\n", dim_1);v
//
//    int dim_0_stride = PyArray_STRIDE(vector, 0) / PyArray_ITEMSIZE(vector);
//    printf("dim_0_stride: %d\n", dim_0_stride);
//
//    int dim_1_stride = PyArray_STRIDE(vector, 1) / PyArray_ITEMSIZE(vector);
//    printf("dim_1_stride: %d\n", dim_1_stride);
//
//    int itemsize = PyArray_ITEMSIZE(vector);
//    printf("itemsize: %d\n", itemsize);
//
//    double value;
//    for (int i = 0; i < dim_0; i++){
//        for (int j = 0; j < dim_1; j++){
//            value = *((double *)PyArray_GETPTR2(vector,i, j));
//            printf("(%d, %d) = %f (%d)\n", i, j, value, PyArray_GETPTR2(vector,i, j));
//        }
//    }
//
//    //Iterate through the rows of vector
//    double* i = (double*)PyArray_DATA(vector);
//    for(int j = 0; j < dim_0; j++)
//    {
//        printf("%d: %f %d\n", j, *i, i);
//        i += dim_0_stride;
//    }
//
//    printf("Term: %f\n", term);

    /* Clean up Memory */
    Py_DECREF(patterns);
    Py_XDECREF(densities);

    /* Return None */
//    PyObject *returnObject = Py_BuildValue("d", 42.0);
//    return returnObject;
    Py_INCREF(Py_None);
    return Py_None;

    fail:
    Py_XDECREF(patterns);
    Py_XDECREF(densities);
    return NULL;
}

static PyMethodDef method_table[] = {
        {"standard_gaussian",   kernels_standard_gaussian,   METH_VARARGS,   kernels_standardGaussian_docstring},
        /* Sentinel */
        {NULL,                  NULL,                       0,              NULL}
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