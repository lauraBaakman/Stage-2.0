#include "test_kde.h"

#include "test_parzen.h"
#include "test_mbe.h"
#include "test_sambe.h"

CuSuite *KDEGetSuite() {
	CuSuite *suite = CuSuiteNew();
	
	CuSuiteAddSuite(suite, ParzenGetSuite());
	CuSuiteAddSuite(suite, MBEGetSuite());
	CuSuiteAddSuite(suite, SAMBEGetSuite());

	return suite;
}