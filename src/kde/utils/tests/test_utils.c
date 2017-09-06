#include "test_utils.h"

#include "test_knn.h"
#include "test_gsl_utils.h"
#include "test_eigenvalues.h"

CuSuite *UtilsGetSuite() {
    CuSuite *suite = CuSuiteNew();
    CuSuiteAddSuite(suite, KNNGetSuite());
    CuSuiteAddSuite(suite, GSLUtilsGetSuite());
    CuSuiteAddSuite(suite, EigenValuesGetSuite());
    return suite;
}
