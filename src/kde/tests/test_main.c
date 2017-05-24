#include <stdio.h>

#include "CuTest.h"
#include "test_sambe.h"

CuSuite *StrUtilGetSuite();

void RunAllTests(void) {
	CuString *output = CuStringNew();
	CuSuite *suite = CuSuiteNew();
	
	CuSuiteAddSuite(suite, SAMBEGetSuite());
	
	CuSuiteRun(suite);
	CuSuiteSummary(suite, output);
	CuSuiteDetails(suite, output);
	printf("%s\n", output->buffer);
}

int main (void)
{
	//Disable buffering for stdout
	setbuf(stdout, NULL);
	RunAllTests();
}