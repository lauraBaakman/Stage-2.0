#include <stdio.h>

#include "lib/CuTest.h"
#include "kde/tests/test_sambe.h"
#include "kde/tests/test_parzen.h"
#include "kde/tests/test_mbe.h"

CuSuite *StrUtilGetSuite();

int RunAllTests(void) {
	CuString *output = CuStringNew();
	CuSuite *suite = CuSuiteNew();
	
	CuSuiteAddSuite(suite, SAMBEGetSuite());
	CuSuiteAddSuite(suite, ParzenGetSuite());
	CuSuiteAddSuite(suite, MBEGetSuite());

    int exitCode = CuSuiteRun(suite);
	CuSuiteSummary(suite, output);
	CuSuiteDetails(suite, output);
	printf("%s\n", output->buffer);

	CuSuiteDelete(suite);
	CuStringDelete(output);

    return exitCode;
}

int main (void)
{
	//Disable buffering for stdout
	setbuf(stdout, NULL);

	int result =  RunAllTests();

	return result;
}