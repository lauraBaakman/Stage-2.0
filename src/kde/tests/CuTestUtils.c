#include "CuTestUtils.h"
#include <math.h>
#include <gsl/gsl_matrix.h>

void CuAssertMatrixEquals_LineMsg(CuTest* tc,
                                  const char* file, int line, const char* message,
                                  gsl_matrix* expected, gsl_matrix* actual, double delta){
    char buf[STRING_MAX];

    if ((expected->size1 != actual->size1) || (expected->size2 != actual->size2)){
        sprintf(buf, "expected (%d, %d) has a different size than actual (%d, %d)",
                expected->size1, expected->size2,
                actual->size1, actual->size2);
        CuFail_Line(tc, file, line, message, buf);
        return;
    }

    double expectedElement, actualElement;

    for(size_t i = 0; i < actual->size1; i++){
        for(size_t j = 0; j < actual->size2; j++){
            expectedElement = gsl_matrix_get(expected, i, j);
            actualElement = gsl_matrix_get(actual, i, j);

            if (fabs(expectedElement - actualElement) > delta){
                sprintf(buf, "expected <%f> at (%d, %d) but was <%f>", expectedElement, i, j, actualElement);
                CuFail_Line(tc, file, line, message, buf);
            }
        }
    }


}