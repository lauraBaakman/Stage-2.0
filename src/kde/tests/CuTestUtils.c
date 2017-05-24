#include "CuTestUtils.h"
#include <math.h>

void CuAssertMatrixEquals_LineMsg(CuTest* tc,
                                  const char* file, int line, const char* message,
                                  gsl_matrix* expected, gsl_matrix* actual, double delta){
    char buf[STRING_MAX];

    if ((expected->size1 != actual->size1) || (expected->size2 != actual->size2)){
        sprintf(buf, "expected (%lu, %lu) has a different size than actual (%lu, %lu)",
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

            if ((fabs(expectedElement - actualElement) > delta) || (isnan(actualElement))){
                sprintf(buf, "expected <%f> at (%lu, %lu) but was <%f>", expectedElement, i, j, actualElement);
                CuFail_Line(tc, file, line, message, buf);
            }
        }
    }


}

void CuAssertVectorEquals_LineMsg(CuTest *tc, const char *file, int line, const char *message, gsl_vector *expected,
                                  gsl_vector *actual, double delta) {
    char buf[STRING_MAX];
    if (expected->size != actual->size){
        sprintf(buf, "expected (%lu) has a different size than actual (%lu)",
                expected->size, actual->size);
        CuFail_Line(tc, file, line, message, buf);
        return;
    }

    double expectedElement, actualElement;

    for(size_t i = 0; i < actual->size; i++){
        expectedElement = gsl_vector_get(expected, i);
        actualElement = gsl_vector_get(actual, i);

        if ((fabs(expectedElement - actualElement) > delta) || (isnan(actualElement))){
            sprintf(buf, "expected <%f> at (%lu) but was <%f>", expectedElement, i, actualElement);
            CuFail_Line(tc, file, line, message, buf);
        }
    }
}
