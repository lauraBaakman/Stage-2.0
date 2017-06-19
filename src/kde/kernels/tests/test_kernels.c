#include "test_kernels.h"

#include "test_epanechnikov.h"
#include "test_gaussian.h"

CuSuite* KernelsGetSuite(){
	CuSuite *suite = CuSuiteNew();
	CuSuiteAddSuite(suite, EpanechnikovGetSuite());
	CuSuiteAddSuite(suite, GaussianGetSuite());
	return suite;	
}