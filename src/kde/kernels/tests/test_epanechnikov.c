#include "test_epanechnikov.h"

#include <omp.h>

#include "../../../lib/CuTestUtils.h"
#include "../../../test_constants.h"

#include "../epanechnikov.h"
#include "../kernels.h"
#include "../../utils/gsl_utils.h"


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
	gsl_vector_set(expected, 0, 0.577297133446585);

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
	gsl_vector_set(pattern, 0, 0.5);

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
	gsl_vector_set(expected, 0, 0.287224936361155);

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
	gsl_vector_set(expected, 0, 0.462693261119970);
	gsl_vector_set(expected, 1, 0.287224936361155);
	gsl_vector_set(expected, 2, 0.0);
	gsl_vector_set(expected, 3, 0.462693261119970);
	gsl_vector_set(expected, 4, 0.287224936361155);	
	gsl_vector_set(expected, 5, 0.254324625468877);	
	gsl_vector_set(expected, 6, 0.265291395766303);	

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
	gsl_vector_set(expected, 0, 0.462693261119970);
	gsl_vector_set(expected, 1, 0.287224936361155);
	gsl_vector_set(expected, 2, 0.0);
	gsl_vector_set(expected, 3, 0.462693261119970);
	gsl_vector_set(expected, 4, 0.287224936361155);	
	gsl_vector_set(expected, 5, 0.254324625468877);	
	gsl_vector_set(expected, 6, 0.265291395766303);	

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

	#pragma omp parallel
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
	gsl_vector_set(expected, 0, 0.462693261119970);
	gsl_vector_set(expected, 1, 0.287224936361155);
	gsl_vector_set(expected, 2, 0.0);
	gsl_vector_set(expected, 3, 0.462693261119970);
	gsl_vector_set(expected, 4, 0.287224936361155);	
	gsl_vector_set(expected, 5, 0.254324625468877);	
	gsl_vector_set(expected, 6, 0.265291395766303);	

	gsl_vector* actual = gsl_vector_alloc(numPatterns);

	#pragma omp parallel shared(actual, patterns)
	{
		int pid = omp_get_thread_num();
		gsl_vector_view pattern;
		double density;

		#pragma omp parallel for
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

void testSymmetricEpanechnikovMultipleParallel2(CuTest *tc){
	size_t dimension = 2;
	size_t numPatterns = 300;
	size_t numSmallPatterns = 7;
	int numThreads = 1;

	#pragma omp parallel
	{
		numThreads = omp_get_num_threads();
	}

	SymmetricKernel kernel = selectSymmetricKernel(EPANECHNIKOV);
	kernel.prepare(dimension, numThreads);

	gsl_matrix* smallPatterns = gsl_matrix_alloc(numSmallPatterns, dimension);
	gsl_matrix_set(smallPatterns, 0, 0, 0.3); gsl_matrix_set(smallPatterns, 0, 1, 0.5);
	gsl_matrix_set(smallPatterns, 1, 0, 0.5); gsl_matrix_set(smallPatterns, 1, 1, 0.5);
	gsl_matrix_set(smallPatterns, 2, 0, 5.0); gsl_matrix_set(smallPatterns, 2, 1, 0.5);
	gsl_matrix_set(smallPatterns, 3, 0, 0.3); gsl_matrix_set(smallPatterns, 3, 1, 0.5);
	gsl_matrix_set(smallPatterns, 4, 0, 0.5); gsl_matrix_set(smallPatterns, 4, 1, 0.5);
	gsl_matrix_set(smallPatterns, 5, 0, 0.2); gsl_matrix_set(smallPatterns, 5, 1, 0.7);	
	gsl_matrix_set(smallPatterns, 6, 0, 0.4); gsl_matrix_set(smallPatterns, 6, 1, 0.6);	

	gsl_vector* smallExpected = gsl_vector_alloc(numSmallPatterns);
	gsl_vector_set(smallExpected, 0, 0.462693261119970);
	gsl_vector_set(smallExpected, 1, 0.287224936361155);
	gsl_vector_set(smallExpected, 2, 0.0);
	gsl_vector_set(smallExpected, 3, 0.462693261119970);
	gsl_vector_set(smallExpected, 4, 0.287224936361155);	
	gsl_vector_set(smallExpected, 5, 0.254324625468877);	
	gsl_vector_set(smallExpected, 6, 0.265291395766303);	


	gsl_matrix* patterns = gsl_matrix_alloc(numPatterns, dimension);
	gsl_vector* expected = gsl_vector_alloc(numPatterns);

	for(size_t i = 0, smallI = -1; i < numPatterns; i++){
		smallI = (smallI + 1) % numSmallPatterns;
		gsl_matrix_set(patterns, i, 0, gsl_matrix_get(smallPatterns, smallI, 0));
		gsl_matrix_set(patterns, i, 1, gsl_matrix_get(smallPatterns, smallI, 1));
		gsl_vector_set(expected, i, gsl_vector_get(smallExpected, smallI));
	}

	gsl_vector* actual = gsl_vector_alloc(numPatterns);

	#pragma omp parallel shared(actual, patterns)
	{
		int pid = omp_get_thread_num();
		gsl_vector_view pattern;
		double density;

		#pragma omp parallel for
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
	gsl_matrix_free(smallPatterns);
	gsl_vector_free(smallExpected);
	kernel.free();
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
    SUITE_ADD_TEST(suite, testSymmetricEpanechnikovMultipleParallel2);
    return suite;	
}