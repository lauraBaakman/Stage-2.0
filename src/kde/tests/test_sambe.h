#ifndef __kde__test_sambe__
#define __kde__test_sambe__

#include <stdio.h>

#include "../../lib/CuTest.h"
#include "../../lib/CuTestUtils.h"

#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_matrix.h>

#include "../../test_constants.h"
#include "../sambe.h"

CuSuite* SAMBEGetSuite();

#endif