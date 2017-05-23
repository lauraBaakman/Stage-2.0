#ifndef CUTESTUTILS_C_H
#define CUTESTUTILS_C_H

#include "CuTest.h"
#include <gsl/gsl_matrix.h>
#include <stdbool.h>

#ifndef isnan
inline bool isnan(double x) {
    return x != x;
}
#endif

void CuAssertMatrixEquals_LineMsg(CuTest* tc,
                               const char* file, int line, const char* message,
                               gsl_matrix* expected, gsl_matrix* actual, double delta);

#define CuAssertMatrixEquals(tc,ex,ac,dl) CuAssertMatrixEquals_LineMsg((tc),__FILE__,__LINE__,NULL,(ex),(ac),(dl))


#endif
