#include "test_kernels.h"

#include "test_epanechnikov.h"

CuSuite* KernelsGetSuite(){
	CuSuite *suite = CuSuiteNew();
	CuSuiteAddSuite(suite, EpanechnikovGetSuite());
	return suite;	
}