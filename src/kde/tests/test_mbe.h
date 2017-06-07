#ifndef __kde__test_mbe__
#define __kde__test_mbe__

#include <stdio.h>

#include "../../lib/CuTest.h"
#include "../../lib/CuTestUtils.h"

#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_matrix.h>

#include "../../test_constants.h"
#include "../mbe.h"

CuSuite* MBEGetSuite();

#endif