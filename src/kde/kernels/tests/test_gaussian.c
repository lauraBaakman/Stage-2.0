#include "test_epanechnikov.h"

#include <omp.h>

#include "../../../lib/CuTestUtils.h"
#include "../../../test_utils.h"

#include "../gaussian.h"
#include "../kernels.h"
#include "../../utils/gsl_utils.h"

static int g_numThreads = 2;

void testSymmetricGaussianSingle(CuTest *tc){
	size_t dimension = 2;
	size_t numPatterns = 1;
	int numThreads = 1;

	SymmetricKernel kernel = selectSymmetricKernel(STANDARD_GAUSSIAN);
	kernel.prepare(dimension, numThreads);

	gsl_vector* pattern = gsl_vector_alloc(dimension);
	gsl_vector_set(pattern, 0, 0.5);
	gsl_vector_set(pattern, 1, 0.5);

	gsl_vector* expected = gsl_vector_alloc(numPatterns);
	gsl_vector_set(expected, 0, 0.123949994309653);

	gsl_vector* actual = gsl_vector_alloc(numPatterns);
	gsl_vector_set(actual, 0, 
		kernel.density(pattern, 0)
	);	

	CuAssertVectorEquals(tc, expected, actual, delta);

	gsl_vector_free(actual);
	gsl_vector_free(pattern);
	gsl_vector_free(expected);
	kernel.free();
}

void testSymmetricGaussianMultipleSingleThreaded(CuTest *tc){
	size_t dimension = 3;
	size_t numPatterns = 3;
	int numThreads = 1;

	SymmetricKernel kernel = selectSymmetricKernel(STANDARD_GAUSSIAN);
	kernel.prepare(dimension, numThreads);

	gsl_matrix* patterns = gsl_matrix_alloc(numPatterns, dimension);
	gsl_matrix_set(patterns, 0, 0, +0.00); gsl_matrix_set(patterns, 0, 1, +0.0); gsl_matrix_set(patterns, 0, 2, 0.0);
	gsl_matrix_set(patterns, 1, 0, +0.50); gsl_matrix_set(patterns, 1, 1, +0.5); gsl_matrix_set(patterns, 1, 2, 0.5);
	gsl_matrix_set(patterns, 2, 0, -0.75); gsl_matrix_set(patterns, 2, 1, -0.5); gsl_matrix_set(patterns, 2, 2, 0.1);

	gsl_vector* expected = gsl_vector_alloc(numPatterns);
	gsl_vector_set(expected, 0, 0.063493635934241);
	gsl_vector_set(expected, 1, 0.043638495249061);
	gsl_vector_set(expected, 2, 0.042084928316873);

	gsl_vector* actual = gsl_vector_alloc(numPatterns);
	
	gsl_vector_view pattern;
	double density;
	for(size_t i = 0; i < numPatterns; i++){
		pattern = gsl_matrix_row(patterns, i);
		density = kernel.density(&pattern.vector, 0);
		gsl_vector_set(actual, i, density);
	}

	CuAssertVectorEquals(tc, actual, expected, delta);

	gsl_vector_free(actual);
	gsl_matrix_free(patterns);
	gsl_vector_free(expected);
	kernel.free();
}

void testSymmetricGaussianMultipleParallel(CuTest *tc){
	size_t dimension = 3;
	size_t numPatterns = 3;
	
	int numThreads = 1;
	#pragma omp parallel num_threads(g_numThreads)
	{
		numThreads = omp_get_num_threads();
	}

	SymmetricKernel kernel = selectSymmetricKernel(STANDARD_GAUSSIAN);
	kernel.prepare(dimension, numThreads);

	gsl_matrix* patterns = gsl_matrix_alloc(numPatterns, dimension);
	gsl_matrix_set(patterns, 0, 0, +0.00); gsl_matrix_set(patterns, 0, 1, +0.0); gsl_matrix_set(patterns, 0, 2, 0.0);
	gsl_matrix_set(patterns, 1, 0, +0.50); gsl_matrix_set(patterns, 1, 1, +0.5); gsl_matrix_set(patterns, 1, 2, 0.5);
	gsl_matrix_set(patterns, 2, 0, -0.75); gsl_matrix_set(patterns, 2, 1, -0.5); gsl_matrix_set(patterns, 2, 2, 0.1);

	gsl_vector* expected = gsl_vector_alloc(numPatterns);
	gsl_vector_set(expected, 0, 0.063493635934241);
	gsl_vector_set(expected, 1, 0.043638495249061);
	gsl_vector_set(expected, 2, 0.042084928316873);

	gsl_vector* actual = gsl_vector_alloc(numPatterns);
	

	#pragma omp parallel shared(patterns, actual) num_threads(g_numThreads)
	{
		gsl_vector_view pattern;
		double density;
		int pid = omp_get_thread_num();

		#pragma omp for
		for(size_t i = 0; i < numPatterns; i++)
		{
			pattern = gsl_matrix_row(patterns, i);
			density = kernel.density(&pattern.vector, pid);
			gsl_vector_set(actual, i, density);
		}

	}

	CuAssertVectorEquals(tc, actual, expected, delta);

	gsl_vector_free(actual);
	gsl_matrix_free(patterns);
	gsl_vector_free(expected);
	kernel.free();
}

void testSAGaussianSingle(CuTest *tc){
	size_t dimension = 3;
	size_t numPatterns = 1;

	int numThreads = 1;

	ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(SHAPE_ADAPTIVE_GAUSSIAN);

	gsl_matrix* H = gsl_matrix_alloc(dimension, dimension);
	gsl_matrix_set(H, 0, 0, +2); gsl_matrix_set(H, 0, 1, -1); gsl_matrix_set(H, 0, 2, +0);
	gsl_matrix_set(H, 1, 0, -1); gsl_matrix_set(H, 1, 1, +2); gsl_matrix_set(H, 1, 2, -1);
	gsl_matrix_set(H, 2, 0, +0); gsl_matrix_set(H, 2, 1, -1); gsl_matrix_set(H, 2, 2, +2);

	gsl_vector* pattern = gsl_vector_alloc(dimension);
	gsl_vector_set(pattern, 0, 0.05);
	gsl_vector_set(pattern, 1, 0.05);
	gsl_vector_set(pattern, 2, 0.05);

	gsl_vector* expected = gsl_vector_alloc(numPatterns);
	gsl_vector_set(expected, 0, 0.121703390601269);

	gsl_vector* actual = gsl_vector_alloc(numPatterns);

	int pid = 0;
	kernel.allocate(dimension, numThreads);
	kernel.computeConstants(H, pid);

	double localBandwidth = 0.5;
	gsl_vector_set(actual, 0, 
		kernel.density(pattern, localBandwidth, pid)
	);

	CuAssertVectorEquals(tc, expected, actual, delta);

	gsl_vector_free(actual);
	gsl_vector_free(pattern);
	gsl_vector_free(expected);
	gsl_matrix_free(H);
}

void testSAGaussianMultipleSingleThreaded(CuTest *tc){
	size_t dimension = 3;
	size_t numPatterns = 3;

	int numThreads = 1;

	ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(SHAPE_ADAPTIVE_GAUSSIAN);

	gsl_matrix* H = gsl_matrix_alloc(dimension, dimension);
	gsl_matrix_set(H, 0, 0, +2); gsl_matrix_set(H, 0, 1, -1); gsl_matrix_set(H, 0, 2, +0);
	gsl_matrix_set(H, 1, 0, -1); gsl_matrix_set(H, 1, 1, +2); gsl_matrix_set(H, 1, 2, -1);
	gsl_matrix_set(H, 2, 0, +0); gsl_matrix_set(H, 2, 1, -1); gsl_matrix_set(H, 2, 2, +2);

	gsl_matrix* patterns = gsl_matrix_alloc(numPatterns, dimension);
	gsl_matrix_set(patterns, 0, 0, 0.05); gsl_matrix_set(patterns, 0, 1, 0.05); gsl_matrix_set(patterns, 0, 2, 0.05);
	gsl_matrix_set(patterns, 1, 0, 0.02); gsl_matrix_set(patterns, 1, 1, 0.03); gsl_matrix_set(patterns, 1, 2, 0.04);
	gsl_matrix_set(patterns, 2, 0, 0.04); gsl_matrix_set(patterns, 2, 1, 0.05); gsl_matrix_set(patterns, 2, 2, 0.03);	

	gsl_vector* expected = gsl_vector_alloc(numPatterns);
	gsl_vector_set(expected, 0, 0.121703390601269);
	gsl_vector_set(expected, 1, 0.045915970935366);
	gsl_vector_set(expected, 2, 1.656546521485471);

	gsl_vector* localBandwidths = gsl_vector_alloc(numPatterns);
	gsl_vector_set(localBandwidths, 0, 0.5);
	gsl_vector_set(localBandwidths, 1, 0.7);
	gsl_vector_set(localBandwidths, 2, 0.2);	

	gsl_vector* actual = gsl_vector_alloc(numPatterns);

	int pid = 0;
	kernel.allocate(dimension, numThreads);
	kernel.computeConstants(H, pid);

	double density, localBandwidth;
	gsl_vector_view pattern;
	for(size_t i = 0; i < patterns->size1; i++){
		localBandwidth = gsl_vector_get(localBandwidths, i);
		pattern = gsl_matrix_row(patterns, i);
		density = kernel.density(&pattern.vector, localBandwidth, pid);
		gsl_vector_set(actual, i, density);
	}

	CuAssertVectorEquals(tc, expected, actual, delta);

	gsl_vector_free(actual);
	gsl_vector_free(expected);
	gsl_vector_free(localBandwidths);
	gsl_matrix_free(patterns);
	gsl_matrix_free(H);
}

void testSAGaussianMultipleParallel(CuTest *tc){
	size_t dimension = 3;
	size_t numPatterns = 3;

	int numThreads = 1;
	#pragma omp parallel num_threads(g_numThreads)
	{
		numThreads = omp_get_num_threads();
	}

	ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(SHAPE_ADAPTIVE_GAUSSIAN);

	gsl_matrix* H = gsl_matrix_alloc(dimension, dimension);
	gsl_matrix_set(H, 0, 0, +2); gsl_matrix_set(H, 0, 1, -1); gsl_matrix_set(H, 0, 2, +0);
	gsl_matrix_set(H, 1, 0, -1); gsl_matrix_set(H, 1, 1, +2); gsl_matrix_set(H, 1, 2, -1);
	gsl_matrix_set(H, 2, 0, +0); gsl_matrix_set(H, 2, 1, -1); gsl_matrix_set(H, 2, 2, +2);

	gsl_matrix* patterns = gsl_matrix_alloc(numPatterns, dimension);
	gsl_matrix_set(patterns, 0, 0, 0.05); gsl_matrix_set(patterns, 0, 1, 0.05); gsl_matrix_set(patterns, 0, 2, 0.05);
	gsl_matrix_set(patterns, 1, 0, 0.02); gsl_matrix_set(patterns, 1, 1, 0.03); gsl_matrix_set(patterns, 1, 2, 0.04);
	gsl_matrix_set(patterns, 2, 0, 0.04); gsl_matrix_set(patterns, 2, 1, 0.05); gsl_matrix_set(patterns, 2, 2, 0.03);	

	gsl_vector* expected = gsl_vector_alloc(numPatterns);
	gsl_vector_set(expected, 0, 0.121703390601269);
	gsl_vector_set(expected, 1, 0.045915970935366);
	gsl_vector_set(expected, 2, 1.656546521485471);

	gsl_vector* localBandwidths = gsl_vector_alloc(numPatterns);
	gsl_vector_set(localBandwidths, 0, 0.5);
	gsl_vector_set(localBandwidths, 1, 0.7);
	gsl_vector_set(localBandwidths, 2, 0.2);	

	gsl_vector* actual = gsl_vector_alloc(numPatterns);

	kernel.allocate(dimension, numThreads);

	#pragma omp parallel num_threads(g_numThreads) shared(actual, patterns, kernel, H, numThreads, dimension, localBandwidths) 
	{
		int pid = omp_get_thread_num();

		kernel.computeConstants(H, pid);

		double density, localBandwidth;
		gsl_vector_view pattern;
		
		#pragma omp parallel
		for(size_t i = 0; i < patterns->size1; i++){
			localBandwidth = gsl_vector_get(localBandwidths, i);
			pattern = gsl_matrix_row(patterns, i);
			density = kernel.density(&pattern.vector, localBandwidth, pid);
			gsl_vector_set(actual, i, density);
		}
	}

	CuAssertVectorEquals(tc, expected, actual, delta);

	gsl_vector_free(actual);
	gsl_vector_free(expected);
	gsl_vector_free(localBandwidths);
	gsl_matrix_free(patterns);
	gsl_matrix_free(H);
}

CuSuite *GaussianGetSuite(){
    CuSuite *suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, testSymmetricGaussianSingle);
    SUITE_ADD_TEST(suite, testSymmetricGaussianMultipleSingleThreaded);
    SUITE_ADD_TEST(suite, testSymmetricGaussianMultipleParallel);

    SUITE_ADD_TEST(suite, testSAGaussianSingle);
    SUITE_ADD_TEST(suite, testSAGaussianMultipleSingleThreaded);
    SUITE_ADD_TEST(suite, testSAGaussianMultipleParallel);
    return suite;	
}