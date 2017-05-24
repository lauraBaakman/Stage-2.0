#include "test_sambe.h"

void testDetermineGlobalKernelShape(CuTest* tc){
    gsl_matrix* xs = gsl_matrix_alloc(4, 2);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 1, 0, 0); gsl_matrix_set(xs, 1, 1, 1);
    gsl_matrix_set(xs, 2, 0, 1); gsl_matrix_set(xs, 2, 1, 0);
    gsl_matrix_set(xs, 3, 0, 1); gsl_matrix_set(xs, 3, 1, 1);

    gsl_vector* localBandwidths = gsl_vector_alloc(4);
    gsl_vector_set(localBandwidths, 0, 0.840896194313949);
    gsl_vector_set(localBandwidths, 1, 1.189207427458816);
    gsl_vector_set(localBandwidths, 2, 1.189207427458816);
    gsl_vector_set(localBandwidths, 3, 0.840896194313949);

    double globalBandwidth = 0.721347520444482;

    ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(SHAPE_ADAPTIVE_GAUSSIAN);

    int k = 3;

    prepareGlobals(xs, localBandwidths, globalBandwidth, kernel, k);

    size_t pattern_idx = 0;
    determineGlobalKernelShape(pattern_idx);

    gsl_matrix* expected = gsl_matrix_alloc(2, 2);
    gsl_matrix_set(expected, 0, 0, 0.832940370215782); gsl_matrix_set(expected, 0, 1, -0.416470185107891);
    gsl_matrix_set(expected, 1, 0, - 0.416470185107891); gsl_matrix_set(expected, 1, 1, 0.832940370215782);

    gsl_matrix* actual = g_globalBandwidthMatrix;

    CuAssertMatrixEquals(tc, expected, actual, delta);

    gsl_matrix_free(xs);
    gsl_vector_free(localBandwidths);
    freeGlobals();
}

void testPrepareShapeAdaptiveKernelInverse(CuTest* tc){
    gsl_matrix* xs = gsl_matrix_alloc(4, 2);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 1, 0, 0); gsl_matrix_set(xs, 1, 1, 1);
    gsl_matrix_set(xs, 2, 0, 1); gsl_matrix_set(xs, 2, 1, 0);
    gsl_matrix_set(xs, 3, 0, 1); gsl_matrix_set(xs, 3, 1, 1);

    gsl_vector* localBandwidths = gsl_vector_alloc(4);
    gsl_vector_set(localBandwidths, 0, 0.840896194313949);
    gsl_vector_set(localBandwidths, 1, 1.189207427458816);
    gsl_vector_set(localBandwidths, 2, 1.189207427458816);
    gsl_vector_set(localBandwidths, 3, 0.840896194313949);

    double globalBandwidth = 0.721347520444482;

    ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(SHAPE_ADAPTIVE_GAUSSIAN);

    int k = 3;

    prepareGlobals(xs, localBandwidths, globalBandwidth, kernel, k);

    size_t pattern_idx = 0;
    prepareShapeAdaptiveKernel(pattern_idx);

    /* Check the inverse */
    gsl_matrix* expected_inverse = gsl_matrix_alloc(2, 2);
    gsl_matrix_set(expected_inverse, 0, 0, 1.600754845137257); gsl_matrix_set(expected_inverse, 0, 1, 0.800377422568628);
    gsl_matrix_set(expected_inverse, 1, 0, 0.800377422568628); gsl_matrix_set(expected_inverse, 1, 1, 1.600754845137257);

    gsl_matrix* actual_inverse = g_globalInverse;

    CuAssertMatrixEquals(tc, expected_inverse, actual_inverse, delta);

    gsl_matrix_free(xs);
    gsl_vector_free(localBandwidths);
    freeGlobals();
}

void testPrepareShapeAdaptiveKernelScalingFactor(CuTest* tc){
    gsl_matrix* xs = gsl_matrix_alloc(4, 2);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 1, 0, 0); gsl_matrix_set(xs, 1, 1, 1);
    gsl_matrix_set(xs, 2, 0, 1); gsl_matrix_set(xs, 2, 1, 0);
    gsl_matrix_set(xs, 3, 0, 1); gsl_matrix_set(xs, 3, 1, 1);

    gsl_vector* localBandwidths = gsl_vector_alloc(4);
    gsl_vector_set(localBandwidths, 0, 0.840896194313949);
    gsl_vector_set(localBandwidths, 1, 1.189207427458816);
    gsl_vector_set(localBandwidths, 2, 1.189207427458816);
    gsl_vector_set(localBandwidths, 3, 0.840896194313949);

    double globalBandwidth = 0.721347520444482;

    ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(SHAPE_ADAPTIVE_GAUSSIAN);

    int k = 3;

    prepareGlobals(xs, localBandwidths, globalBandwidth, kernel, k);

    size_t pattern_idx = 0;
    prepareShapeAdaptiveKernel(pattern_idx);

    /* Check the Scaling Factor */
    double expected_scaling_factor = 1.921812055672802;
    double actual_scaling_factor = g_globalScalingFactor;

    CuAssertDblEquals(tc, expected_scaling_factor, actual_scaling_factor, delta);

    gsl_matrix_free(xs);
    gsl_vector_free(localBandwidths);
    freeGlobals();
}

void testPrepareShapeAdaptiveKernelPDFConstant(CuTest* tc){
    gsl_matrix* xs = gsl_matrix_alloc(4, 2);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 1, 0, 0); gsl_matrix_set(xs, 1, 1, 1);
    gsl_matrix_set(xs, 2, 0, 1); gsl_matrix_set(xs, 2, 1, 0);
    gsl_matrix_set(xs, 3, 0, 1); gsl_matrix_set(xs, 3, 1, 1);

    gsl_vector* localBandwidths = gsl_vector_alloc(4);
    gsl_vector_set(localBandwidths, 0, 0.840896194313949);
    gsl_vector_set(localBandwidths, 1, 1.189207427458816);
    gsl_vector_set(localBandwidths, 2, 1.189207427458816);
    gsl_vector_set(localBandwidths, 3, 0.840896194313949);

    double globalBandwidth = 0.721347520444482;

    ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(SHAPE_ADAPTIVE_GAUSSIAN);

    int k = 3;

    prepareGlobals(xs, localBandwidths, globalBandwidth, kernel, k);

    size_t pattern_idx = 0;
    prepareShapeAdaptiveKernel(pattern_idx);

    /* Check the pdf Constant */
    double expected_pdf_constant = 0.159154943091895;
    double actual_pdf_constant = g_kernelConstant;

    CuAssertDblEquals(tc, expected_pdf_constant, actual_pdf_constant, delta);

    gsl_matrix_free(xs);
    gsl_vector_free(localBandwidths);
    freeGlobals();
}

void testEvaluateKernel(CuTest* tc){
    double localBandwidth = 0.5;

    gsl_vector* x = gsl_vector_alloc(3);
    gsl_vector_set(x, 0, 0.05);
    gsl_vector_set(x, 1, 0.05);
    gsl_vector_set(x, 2, 0.05);
    gsl_vector* x_original = gsl_vector_alloc(3);
    gsl_vector_memcpy(x_original, x);

    gsl_vector* xi = gsl_vector_alloc(3);
    gsl_vector_set(xi, 0, 0.02);
    gsl_vector_set(xi, 1, 0.03);
    gsl_vector_set(xi, 2, 0.04);
    gsl_vector* xi_original = gsl_vector_alloc(3);
    gsl_vector_memcpy(xi_original, xi);

    allocateGlobals(3, 1, 1);
    gsl_matrix_set(g_globalBandwidthMatrix, 0, 0, 2); gsl_matrix_set(g_globalBandwidthMatrix, 0, 1, -1); gsl_matrix_set(g_globalBandwidthMatrix, 0, 2, 0);
    gsl_matrix_set(g_globalBandwidthMatrix, 1, 0, -1); gsl_matrix_set(g_globalBandwidthMatrix, 1, 1, 2); gsl_matrix_set(g_globalBandwidthMatrix, 1, 2, -1);
    gsl_matrix_set(g_globalBandwidthMatrix, 2, 0, 0); gsl_matrix_set(g_globalBandwidthMatrix, 2, 1, -1); gsl_matrix_set(g_globalBandwidthMatrix, 2, 2, 2);

    gsl_matrix_set(g_globalInverse, 0, 0, 0.75); gsl_matrix_set(g_globalInverse, 0, 1, 0.50); gsl_matrix_set(g_globalInverse, 0, 2, 0.25);
    gsl_matrix_set(g_globalInverse, 1, 0, 0.50); gsl_matrix_set(g_globalInverse, 1, 1, 1.00); gsl_matrix_set(g_globalInverse, 1, 2, 0.50);
    gsl_matrix_set(g_globalInverse, 2, 0, 0.25); gsl_matrix_set(g_globalInverse, 2, 1, 0.50); gsl_matrix_set(g_globalInverse, 2, 2, 0.75);

    g_globalScalingFactor = 0.25;
    g_kernelConstant = 0.063493635934241;

    double actual = evaluateKernel(x, xi, localBandwidth);

    double expected = 0.126114075683830;


    CuAssertDblEquals(tc, expected, actual, delta);

    /* Test if x and xi are unchanged */
    CuAssertVectorEquals(tc, x_original, x, delta);
    CuAssertVectorEquals(tc, xi_original, xi, delta);

    freeGlobals();
    gsl_vector_free(x);
    gsl_vector_free(xi);
}

CuSuite *SAMBEGetSuite() {
	CuSuite *suite = CuSuiteNew();
	SUITE_ADD_TEST(suite, testDetermineGlobalKernelShape);
    SUITE_ADD_TEST(suite, testPrepareShapeAdaptiveKernelInverse);
    SUITE_ADD_TEST(suite, testPrepareShapeAdaptiveKernelScalingFactor);
    SUITE_ADD_TEST(suite, testPrepareShapeAdaptiveKernelPDFConstant);
    SUITE_ADD_TEST(suite, testEvaluateKernel);
	return suite;
}