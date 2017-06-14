#include <stdio.h>

#include "lib/CuTest.h"
#include "kde/tests/test_kde.h"
#include "kde/utils/tests/test_utils.h"
#include "kde/kernels/tests/test_kernels.h"

CuSuite *StrUtilGetSuite();

int RunAllTests(void) {
	CuString *output = CuStringNew();
	CuSuite *suite = CuSuiteNew();
	
	CuSuiteAddSuite(suite, KDEGetSuite());
	CuSuiteAddSuite(suite, UtilsGetSuite());
	CuSuiteAddSuite(suite, KernelsGetSuite());

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