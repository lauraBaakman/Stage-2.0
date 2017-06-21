#include "test_kde.h"

#include "test_parzen.h"

CuSuite *KDEGetSuite() {
	CuSuite *suite = CuSuiteNew();
	
	CuSuiteAddSuite(suite, ParzenGetSuite());

	return suite;
}