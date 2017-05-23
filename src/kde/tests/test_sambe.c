#include "test_sambe.h"

void testDetermineGlobalKernelShape(CuTest* tc){
    gsl_matrix* xs = gsl_matrix_alloc(4, 2);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 1);
    gsl_matrix_set(xs, 0, 0, 1); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 0, 0, 1); gsl_matrix_set(xs, 0, 1, 1);

    gsl_vector* localBandwidths = gsl_vector_alloc(4);
    gsl_vector_set(localBandwidths, 0, 0.840896194313949);
    gsl_vector_set(localBandwidths, 1, 1.189207427458816);
    gsl_vector_set(localBandwidths, 2, 1.189207427458816);
    gsl_vector_set(localBandwidths, 3, 0.840896194313949);

    double globalBandwidth = 0.721347520444482;

    ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(SHAPE_ADAPTIVE_GAUSSIAN);

    prepareGlobals(xs, localBandwidths, globalBandwidth, kernel);

    size_t pattern_idx = 0;
    determineGlobalKernelShape(pattern_idx);

    gsl_matrix* expected = gsl_matrix_alloc(2, 2);
    gsl_matrix_set(expected, 0, 0, 0.832940370215782); gsl_matrix_set(expected, 0, 1, -0.416470185107891);
    gsl_matrix_set(expected, 1, 1, - 0.416470185107891); gsl_matrix_set(expected, 1, 1, 0.832940370215782);

    gsl_matrix* actual = g_globalBandwidthMatrix;

    CuAssertMatrixEquals(tc, expected, actual, delta);
}

void testPrepareShapeAdaptiveKernelInverse(CuTest* tc){
    gsl_matrix* xs = gsl_matrix_alloc(4, 2);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 1);
    gsl_matrix_set(xs, 0, 0, 1); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 0, 0, 1); gsl_matrix_set(xs, 0, 1, 1);

    gsl_vector* localBandwidths = gsl_vector_alloc(4);
    gsl_vector_set(localBandwidths, 0, 0.840896194313949);
    gsl_vector_set(localBandwidths, 1, 1.189207427458816);
    gsl_vector_set(localBandwidths, 2, 1.189207427458816);
    gsl_vector_set(localBandwidths, 3, 0.840896194313949);

    double globalBandwidth = 0.721347520444482;

    ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(SHAPE_ADAPTIVE_GAUSSIAN);

    prepareGlobals(xs, localBandwidths, globalBandwidth, kernel);

    size_t pattern_idx = 0;
    prepareShapeAdaptiveKernel(pattern_idx);

    /* Check the inverse */
    gsl_matrix* expected_inverse = gsl_matrix_alloc(2, 2);
    gsl_matrix_set(expected_inverse, 0, 0, 1.600754845137257); gsl_matrix_set(expected_inverse, 0, 1, 0.800377422568628);
    gsl_matrix_set(expected_inverse, 1, 1, 0.800377422568628); gsl_matrix_set(expected_inverse, 1, 1, 1.600754845137257);

    gsl_matrix* actual_inverse = g_globalBandwidthMatrixInverse;

    CuAssertMatrixEquals(tc, expected_inverse, actual_inverse, delta);
}

void testPrepareShapeAdaptiveKernelScalingFactor(CuTest* tc){
    gsl_matrix* xs = gsl_matrix_alloc(4, 2);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 1);
    gsl_matrix_set(xs, 0, 0, 1); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 0, 0, 1); gsl_matrix_set(xs, 0, 1, 1);

    gsl_vector* localBandwidths = gsl_vector_alloc(4);
    gsl_vector_set(localBandwidths, 0, 0.840896194313949);
    gsl_vector_set(localBandwidths, 1, 1.189207427458816);
    gsl_vector_set(localBandwidths, 2, 1.189207427458816);
    gsl_vector_set(localBandwidths, 3, 0.840896194313949);

    double globalBandwidth = 0.721347520444482;

    ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(SHAPE_ADAPTIVE_GAUSSIAN);

    prepareGlobals(xs, localBandwidths, globalBandwidth, kernel);

    size_t pattern_idx = 0;
    prepareShapeAdaptiveKernel(pattern_idx);

    /* Check the Scaling Factor */
    double expected_scaling_factor = 1.921812055672802;
    double actual_scaling_factor = g_globalBandwidthMatrixDeterminant;

    CuAssertDblEquals(tc, expected_scaling_factor, actual_scaling_factor, delta);
}

void testPrepareShapeAdaptiveKernelPDFConstant(CuTest* tc){
    gsl_matrix* xs = gsl_matrix_alloc(4, 2);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 0, 0, 0); gsl_matrix_set(xs, 0, 1, 1);
    gsl_matrix_set(xs, 0, 0, 1); gsl_matrix_set(xs, 0, 1, 0);
    gsl_matrix_set(xs, 0, 0, 1); gsl_matrix_set(xs, 0, 1, 1);

    gsl_vector* localBandwidths = gsl_vector_alloc(4);
    gsl_vector_set(localBandwidths, 0, 0.840896194313949);
    gsl_vector_set(localBandwidths, 1, 1.189207427458816);
    gsl_vector_set(localBandwidths, 2, 1.189207427458816);
    gsl_vector_set(localBandwidths, 3, 0.840896194313949);

    double globalBandwidth = 0.721347520444482;

    ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(SHAPE_ADAPTIVE_GAUSSIAN);

    prepareGlobals(xs, localBandwidths, globalBandwidth, kernel);

    size_t pattern_idx = 0;
    prepareShapeAdaptiveKernel(pattern_idx);

    /* Check the pdf Constant */
    double expected_pdf_constant = 0.159154943091895;
    double actual_pdf_constant = g_kernelConstant;

    CuAssertDblEquals(tc, expected_pdf_constant, actual_pdf_constant, delta);
}

CuSuite *SAMBEGetSuite() {
	CuSuite *suite = CuSuiteNew();
	SUITE_ADD_TEST(suite, testDetermineGlobalKernelShape);
    SUITE_ADD_TEST(suite, testPrepareShapeAdaptiveKernelInverse);
    SUITE_ADD_TEST(suite, testPrepareShapeAdaptiveKernelScalingFactor);
    SUITE_ADD_TEST(suite, testPrepareShapeAdaptiveKernelPDFConstant);
	return suite;
}