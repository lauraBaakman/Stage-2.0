#include <stdio.h>

#include "lib/CuTest.h"
#include "kde/tests/test_sambe.h"
#include "kde/tests/test_parzen.h"

CuSuite *StrUtilGetSuite();

int RunAllTests(void) {
	CuString *output = CuStringNew();
	CuSuite *suite = CuSuiteNew();
	
	CuSuiteAddSuite(suite, SAMBEGetSuite());
	CuSuiteAddSuite(suite, ParzenGetSuite());

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

	CuSuiteDelete(SAMBEGetSuite());
	CuSuiteDelete(ParzenGetSuite());

	return result;
}