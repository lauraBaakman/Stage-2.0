#include "test_epanechnikov.h"

#include <omp.h>

#include "../../../lib/CuTestUtils.h"
#include "../../../test_utils.h"

#include "../epanechnikov.h"
#include "../kernels.h"
#include "../../utils/gsl_utils.h"

static int g_numThreads = 2;

void testSymmetricEpanechnikovSingle1D1(CuTest *tc){
	size_t dimension = 1;
	size_t numPatterns = 1;

	SymmetricKernel kernel = selectSymmetricKernel(EPANECHNIKOV);
	kernel.prepare(dimension, 1);

	gsl_vector* pattern = gsl_vector_alloc(dimension);
	gsl_vector_set(pattern, 0, 5);

	gsl_vector* expected = gsl_vector_alloc(numPatterns);
	gsl_vector_set(expected, 0, 0);

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

void testSymmetricEpanechnikovSingle1D2(CuTest *tc){
	size_t dimension = 1;
	size_t numPatterns = 1;

	SymmetricKernel kernel = selectSymmetricKernel(EPANECHNIKOV);
	kernel.prepare(dimension, 1);

	gsl_vector* pattern = gsl_vector_alloc(dimension);
	gsl_vector_set(pattern, 0, -0.5);

	gsl_vector* expected = gsl_vector_alloc(numPatterns);
	gsl_vector_set(expected, 0, 0.318639686793720);

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

void testSymmetricEpanechnikovSingle2D1(CuTest *tc){
	size_t dimension = 2;
	size_t numPatterns = 1;

	SymmetricKernel kernel = selectSymmetricKernel(EPANECHNIKOV);
	kernel.prepare(dimension, 1);

	gsl_vector* pattern = gsl_vector_alloc(dimension);
	gsl_vector_set(pattern, 0, 5);
	gsl_vector_set(pattern, 1, 0.5);

	gsl_vector* expected = gsl_vector_alloc(numPatterns);
	gsl_vector_set(expected, 0, 0);

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

void testSymmetricEpanechnikovSingle2D2(CuTest *tc){
	size_t dimension = 2;
	size_t numPatterns = 1;

	SymmetricKernel kernel = selectSymmetricKernel(EPANECHNIKOV);
	kernel.prepare(dimension, 1);

	gsl_vector* pattern = gsl_vector_alloc(dimension);
	gsl_vector_set(pattern, 0, 0.5);
	gsl_vector_set(pattern, 1, 0.5);

	gsl_vector* expected = gsl_vector_alloc(numPatterns);
	gsl_vector_set(expected, 0, 0.114591559026165);

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

void testSymmetricEpanechnikovMultipleSingleThreaded1(CuTest *tc){
	size_t dimension = 2;
	size_t numPatterns = 7;

	SymmetricKernel kernel = selectSymmetricKernel(EPANECHNIKOV);
	kernel.prepare(dimension, 1);

	gsl_matrix* patterns = gsl_matrix_alloc(numPatterns, dimension);
	gsl_matrix_set(patterns, 0, 0, 0.3); gsl_matrix_set(patterns, 0, 1, 0.5);
	gsl_matrix_set(patterns, 1, 0, 0.5); gsl_matrix_set(patterns, 1, 1, 0.5);
	gsl_matrix_set(patterns, 2, 0, 5.0); gsl_matrix_set(patterns, 2, 1, 0.5);
	gsl_matrix_set(patterns, 3, 0, 0.3); gsl_matrix_set(patterns, 3, 1, 0.5);
	gsl_matrix_set(patterns, 4, 0, 0.5); gsl_matrix_set(patterns, 4, 1, 0.5);
	gsl_matrix_set(patterns, 5, 0, 0.2); gsl_matrix_set(patterns, 5, 1, 0.7);	
	gsl_matrix_set(patterns, 6, 0, 0.4); gsl_matrix_set(patterns, 6, 1, 0.6);	

	gsl_vector* expected = gsl_vector_alloc(numPatterns);
	gsl_vector_set(expected, 0, 0.118665925569317);
	gsl_vector_set(expected, 1, 0.114591559026165);
	gsl_vector_set(expected, 2, 0.0);
	gsl_vector_set(expected, 3, 0.118665925569317);
	gsl_vector_set(expected, 4, 0.114591559026165);	
	gsl_vector_set(expected, 5, 0.113827615299324);	
	gsl_vector_set(expected, 6, 0.114082263208271);	

	gsl_vector* actual = gsl_vector_alloc(numPatterns);


	gsl_vector_view pattern;
	double density;
	for(size_t i = 0; i < numPatterns; i++){
		pattern = gsl_matrix_row(patterns, i);
		density = kernel.density(&pattern.vector, 0);
		gsl_vector_set(actual, i, density);
	}

	CuAssertVectorEquals(tc, actual, expected, delta);

	gsl_vector_free(expected);
	gsl_vector_free(actual);
	gsl_matrix_free(patterns);
	kernel.free();
}

void testSymmetricEpanechnikovMultipleSingleThreaded2(CuTest *tc){
	size_t dimension = 2;
	size_t numPatterns = 7;
	int numThreads = 3;

	SymmetricKernel kernel = selectSymmetricKernel(EPANECHNIKOV);
	kernel.prepare(dimension, numThreads);

	gsl_matrix* patterns = gsl_matrix_alloc(numPatterns, dimension);
	gsl_matrix_set(patterns, 0, 0, 0.3); gsl_matrix_set(patterns, 0, 1, 0.5);
	gsl_matrix_set(patterns, 1, 0, 0.5); gsl_matrix_set(patterns, 1, 1, 0.5);
	gsl_matrix_set(patterns, 2, 0, 5.0); gsl_matrix_set(patterns, 2, 1, 0.5);
	gsl_matrix_set(patterns, 3, 0, 0.3); gsl_matrix_set(patterns, 3, 1, 0.5);
	gsl_matrix_set(patterns, 4, 0, 0.5); gsl_matrix_set(patterns, 4, 1, 0.5);
	gsl_matrix_set(patterns, 5, 0, 0.2); gsl_matrix_set(patterns, 5, 1, 0.7);	
	gsl_matrix_set(patterns, 6, 0, 0.4); gsl_matrix_set(patterns, 6, 1, 0.6);	

	gsl_vector* expected = gsl_vector_alloc(numPatterns);
	gsl_vector_set(expected, 0, 0.118665925569317);
	gsl_vector_set(expected, 1, 0.114591559026165);
	gsl_vector_set(expected, 2, 0.0);
	gsl_vector_set(expected, 3, 0.118665925569317);
	gsl_vector_set(expected, 4, 0.114591559026165);	
	gsl_vector_set(expected, 5, 0.113827615299324);	
	gsl_vector_set(expected, 6, 0.114082263208271);	

	gsl_vector* actual = gsl_vector_alloc(numPatterns);


	gsl_vector_view pattern;
	double density;
	int pid = 0;
	for(size_t i = 0; i < numPatterns; i++, pid++){
		pid = pid % numThreads;
		pattern = gsl_matrix_row(patterns, i);
		density = kernel.density(&pattern.vector, pid);
		gsl_vector_set(actual, i, density);
	}

	CuAssertVectorEquals(tc, actual, expected, delta);

	gsl_vector_free(expected);
	gsl_vector_free(actual);
	gsl_matrix_free(patterns);
	kernel.free();
}

void testSymmetricEpanechnikovMultipleParallel1(CuTest *tc){
	size_t dimension = 2;
	size_t numPatterns = 7;
	int numThreads = 1;

	#pragma omp parallel num_threads(g_numThreads)
	{
		numThreads = omp_get_num_threads();
	}

	SymmetricKernel kernel = selectSymmetricKernel(EPANECHNIKOV);
	kernel.prepare(dimension, numThreads);

	gsl_matrix* patterns = gsl_matrix_alloc(numPatterns, dimension);
	gsl_matrix_set(patterns, 0, 0, 0.3); gsl_matrix_set(patterns, 0, 1, 0.5);
	gsl_matrix_set(patterns, 1, 0, 0.5); gsl_matrix_set(patterns, 1, 1, 0.5);
	gsl_matrix_set(patterns, 2, 0, 5.0); gsl_matrix_set(patterns, 2, 1, 0.5);
	gsl_matrix_set(patterns, 3, 0, 0.3); gsl_matrix_set(patterns, 3, 1, 0.5);
	gsl_matrix_set(patterns, 4, 0, 0.5); gsl_matrix_set(patterns, 4, 1, 0.5);
	gsl_matrix_set(patterns, 5, 0, 0.2); gsl_matrix_set(patterns, 5, 1, 0.7);	
	gsl_matrix_set(patterns, 6, 0, 0.4); gsl_matrix_set(patterns, 6, 1, 0.6);	

	gsl_vector* expected = gsl_vector_alloc(numPatterns);
	gsl_vector_set(expected, 0, 0.118665925569317);
	gsl_vector_set(expected, 1, 0.114591559026165);
	gsl_vector_set(expected, 2, 0.0);
	gsl_vector_set(expected, 3, 0.118665925569317);
	gsl_vector_set(expected, 4, 0.114591559026165);	
	gsl_vector_set(expected, 5, 0.113827615299324);	
	gsl_vector_set(expected, 6, 0.114082263208271);	

	gsl_vector* actual = gsl_vector_alloc(numPatterns);

	#pragma omp parallel shared(actual, patterns) num_threads(g_numThreads)
	{
		int pid = omp_get_thread_num();
		gsl_vector_view pattern;
		double density;

		#pragma omp for
		for(size_t i = 0; i < patterns->size1; i++){	
			pattern = gsl_matrix_row(patterns, i);
			density = kernel.density(&pattern.vector, pid);
			gsl_vector_set(actual, i, density);
		}
	}

	CuAssertVectorEquals(tc, actual, expected, delta);

	gsl_vector_free(expected);
	gsl_vector_free(actual);
	gsl_matrix_free(patterns);
	kernel.free();
}

void testShapeAdaptiveEpanechhnikovSingle(CuTest *tc){
	size_t dimension = 3;
	size_t numPatterns = 1;

	int numThreads = 1;

	ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(SHAPE_ADAPTIVE_EPANECHNIKOV);

	gsl_matrix* H = gsl_matrix_alloc(dimension, dimension);
	gsl_matrix_set(H, 0, 0, +2); gsl_matrix_set(H, 0, 1, -1); gsl_matrix_set(H, 0, 2, +0);
	gsl_matrix_set(H, 1, 0, -1); gsl_matrix_set(H, 1, 1, +2); gsl_matrix_set(H, 1, 2, -1);
	gsl_matrix_set(H, 2, 0, +0); gsl_matrix_set(H, 2, 1, -1); gsl_matrix_set(H, 2, 2, +2);

	gsl_vector* pattern = gsl_vector_alloc(dimension);
	gsl_vector_set(pattern, 0, 0.05);
	gsl_vector_set(pattern, 1, 0.05);
	gsl_vector_set(pattern, 2, 0.05);

	gsl_vector* expected = gsl_vector_alloc(numPatterns);
	gsl_vector_set(expected, 0, 0.104949387026863);

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

void testShapeAdaptiveEpanechhnikovMultipleSingleThreaded(CuTest *tc){
	size_t dimension = 3;
	size_t numPatterns = 3;

	int numThreads = 1;

	ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(SHAPE_ADAPTIVE_EPANECHNIKOV);

	gsl_matrix* H = gsl_matrix_alloc(dimension, dimension);
	gsl_matrix_set(H, 0, 0, +2); gsl_matrix_set(H, 0, 1, -1); gsl_matrix_set(H, 0, 2, +0);
	gsl_matrix_set(H, 1, 0, -1); gsl_matrix_set(H, 1, 1, +2); gsl_matrix_set(H, 1, 2, -1);
	gsl_matrix_set(H, 2, 0, +0); gsl_matrix_set(H, 2, 1, -1); gsl_matrix_set(H, 2, 2, +2);

	gsl_matrix* patterns = gsl_matrix_alloc(numPatterns, dimension);
	gsl_matrix_set(patterns, 0, 0, 0.05); gsl_matrix_set(patterns, 0, 1, 0.05); gsl_matrix_set(patterns, 0, 2, 0.05);
	gsl_matrix_set(patterns, 1, 0, 0.02); gsl_matrix_set(patterns, 1, 1, 0.03); gsl_matrix_set(patterns, 1, 2, 0.04);
	gsl_matrix_set(patterns, 2, 0, 0.04); gsl_matrix_set(patterns, 2, 1, 0.05); gsl_matrix_set(patterns, 2, 2, 0.03);	

	gsl_vector* expected = gsl_vector_alloc(numPatterns);
	gsl_vector_set(expected, 0, 0.104949387026863);
	gsl_vector_set(expected, 1, 0.038786019064700);
	gsl_vector_set(expected, 2, 1.547770745658397);

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

void testShapeAdaptiveEpanechhnikovMultipleParallel1(CuTest *tc){
	size_t dimension = 3;
	size_t numPatterns = 3;

	int numThreads = 1;
	#pragma omp parallel num_threads(g_numThreads)
	{
		numThreads = omp_get_num_threads();
	}

	ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(SHAPE_ADAPTIVE_EPANECHNIKOV);

	gsl_matrix* H = gsl_matrix_alloc(dimension, dimension);
	gsl_matrix_set(H, 0, 0, +2); gsl_matrix_set(H, 0, 1, -1); gsl_matrix_set(H, 0, 2, +0);
	gsl_matrix_set(H, 1, 0, -1); gsl_matrix_set(H, 1, 1, +2); gsl_matrix_set(H, 1, 2, -1);
	gsl_matrix_set(H, 2, 0, +0); gsl_matrix_set(H, 2, 1, -1); gsl_matrix_set(H, 2, 2, +2);

	gsl_matrix* patterns = gsl_matrix_alloc(numPatterns, dimension);
	gsl_matrix_set(patterns, 0, 0, 0.05); gsl_matrix_set(patterns, 0, 1, 0.05); gsl_matrix_set(patterns, 0, 2, 0.05);
	gsl_matrix_set(patterns, 1, 0, 0.02); gsl_matrix_set(patterns, 1, 1, 0.03); gsl_matrix_set(patterns, 1, 2, 0.04);
	gsl_matrix_set(patterns, 2, 0, 0.04); gsl_matrix_set(patterns, 2, 1, 0.05); gsl_matrix_set(patterns, 2, 2, 0.03);	

	gsl_vector* expected = gsl_vector_alloc(numPatterns);
	gsl_vector_set(expected, 0, 0.104949387026863);
	gsl_vector_set(expected, 1, 0.038786019064700);
	gsl_vector_set(expected, 2, 1.547770745658397);

	gsl_vector* localBandwidths = gsl_vector_alloc(numPatterns);
	gsl_vector_set(localBandwidths, 0, 0.5);
	gsl_vector_set(localBandwidths, 1, 0.7);
	gsl_vector_set(localBandwidths, 2, 0.2);	

	gsl_vector* actual = gsl_vector_alloc(numPatterns);

	kernel.allocate(dimension, numThreads);

	#pragma omp parallel shared(actual, patterns) num_threads(g_numThreads)
	{
		int pid = omp_get_thread_num();
		gsl_vector_view pattern;
		double density, localBandwidth;
		kernel.computeConstants(H, pid);

		#pragma omp for
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

void testShapeAdaptiveEpanechhnikovMultipleParallel2(CuTest *tc){
	size_t dimension = 3;
	size_t numSmallPatterns = 3;
	size_t numPatterns = 800;

	int numThreads = 1;
	#pragma omp parallel num_threads(g_numThreads)
	{
		numThreads = omp_get_num_threads();
	}

	ShapeAdaptiveKernel kernel = selectShapeAdaptiveKernel(SHAPE_ADAPTIVE_EPANECHNIKOV);

	gsl_matrix* H = gsl_matrix_alloc(dimension, dimension);
	gsl_matrix_set(H, 0, 0, +2); gsl_matrix_set(H, 0, 1, -1); gsl_matrix_set(H, 0, 2, +0);
	gsl_matrix_set(H, 1, 0, -1); gsl_matrix_set(H, 1, 1, +2); gsl_matrix_set(H, 1, 2, -1);
	gsl_matrix_set(H, 2, 0, +0); gsl_matrix_set(H, 2, 1, -1); gsl_matrix_set(H, 2, 2, +2);

	gsl_matrix* smallPatterns = gsl_matrix_alloc(numSmallPatterns, dimension);
	gsl_matrix_set(smallPatterns, 0, 0, 0.05); gsl_matrix_set(smallPatterns, 0, 1, 0.05); gsl_matrix_set(smallPatterns, 0, 2, 0.05);
	gsl_matrix_set(smallPatterns, 1, 0, 0.02); gsl_matrix_set(smallPatterns, 1, 1, 0.03); gsl_matrix_set(smallPatterns, 1, 2, 0.04);
	gsl_matrix_set(smallPatterns, 2, 0, 0.04); gsl_matrix_set(smallPatterns, 2, 1, 0.05); gsl_matrix_set(smallPatterns, 2, 2, 0.03);	

	gsl_vector* smallExpected = gsl_vector_alloc(numSmallPatterns);
	gsl_vector_set(smallExpected, 0, 0.104949387026863);
	gsl_vector_set(smallExpected, 1, 0.038786019064700);
	gsl_vector_set(smallExpected, 2, 1.547770745658397);

	gsl_vector* smalllocalBandwidths = gsl_vector_alloc(numSmallPatterns);
	gsl_vector_set(smalllocalBandwidths, 0, 0.5);
	gsl_vector_set(smalllocalBandwidths, 1, 0.7);
	gsl_vector_set(smalllocalBandwidths, 2, 0.2);	

	gsl_matrix* patterns = gsl_matrix_alloc(numPatterns, dimension);
	gsl_vector* expected = gsl_vector_alloc(numPatterns);
	gsl_vector* localBandwidths = gsl_vector_alloc(numPatterns);

	for(size_t i = 0, smallI = -1; i < numPatterns; i++){
		smallI = (smallI + 1) % numSmallPatterns;
		gsl_matrix_set(patterns, i, 0, gsl_matrix_get(smallPatterns, smallI, 0));
		gsl_matrix_set(patterns, i, 1, gsl_matrix_get(smallPatterns, smallI, 1));
		gsl_matrix_set(patterns, i, 2, gsl_matrix_get(smallPatterns, smallI, 2));
		gsl_vector_set(expected, i, gsl_vector_get(smallExpected, smallI));
		gsl_vector_set(localBandwidths, i, gsl_vector_get(smalllocalBandwidths, smallI));
	}

	gsl_vector* actual = gsl_vector_alloc(numPatterns);

	kernel.allocate(dimension, numThreads);

	#pragma omp parallel shared(actual, patterns) num_threads(g_numThreads)
	{
		int pid = omp_get_thread_num();
		gsl_vector_view pattern;
		double density, localBandwidth;
		kernel.computeConstants(H, pid);

		#pragma omp for
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
	gsl_matrix_free(smallPatterns);
	gsl_vector_free(smalllocalBandwidths);
	gsl_vector_free(smallExpected);
}

CuSuite *EpanechnikovGetSuite(){
    CuSuite *suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, testSymmetricEpanechnikovSingle1D1);
    SUITE_ADD_TEST(suite, testSymmetricEpanechnikovSingle1D2);
    SUITE_ADD_TEST(suite, testSymmetricEpanechnikovSingle2D1);
    SUITE_ADD_TEST(suite, testSymmetricEpanechnikovSingle2D2);
    SUITE_ADD_TEST(suite, testSymmetricEpanechnikovMultipleSingleThreaded1);
    SUITE_ADD_TEST(suite, testSymmetricEpanechnikovMultipleSingleThreaded2);
    SUITE_ADD_TEST(suite, testSymmetricEpanechnikovMultipleParallel1);

    SUITE_ADD_TEST(suite, testShapeAdaptiveEpanechhnikovSingle);
    SUITE_ADD_TEST(suite, testShapeAdaptiveEpanechhnikovMultipleSingleThreaded);
    SUITE_ADD_TEST(suite, testShapeAdaptiveEpanechhnikovMultipleParallel1);
    SUITE_ADD_TEST(suite, testShapeAdaptiveEpanechhnikovMultipleParallel2);

    return suite;
}