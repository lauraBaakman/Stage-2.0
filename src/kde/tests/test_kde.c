#include "test_kde.h"

#include "test_sambe.h"
#include "test_parzen.h"
#include "test_mbe.h"

CuSuite *KDEGetSuite() {
	CuSuite *suite = CuSuiteNew();
	
	CuSuiteAddSuite(suite, SAMBEGetSuite());
	CuSuiteAddSuite(suite, ParzenGetSuite());
	CuSuiteAddSuite(suite, MBEGetSuite());

	return suite;
}