#include "test_kernels.h"

CuSuite* KernelsGetSuite(){
	CuSuite *suite = CuSuiteNew();
	return suite;	
}