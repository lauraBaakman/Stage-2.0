#include <stdio.h>

#include "lib/CuTest.h"
#include "kde/tests/test_sambe.h"

CuSuite *StrUtilGetSuite();

int RunAllTests(void) {
	CuString *output = CuStringNew();
	CuSuite *suite = CuSuiteNew();
	
	CuSuiteAddSuite(suite, SAMBEGetSuite());

    int exitCode = CuSuiteRun(suite);
	CuSuiteSummary(suite, output);
	CuSuiteDetails(suite, output);
	printf("%s\n", output->buffer);

    return exitCode;
}

int main (void)
{
	//Disable buffering for stdout
	setbuf(stdout, NULL);

	return RunAllTests();
}