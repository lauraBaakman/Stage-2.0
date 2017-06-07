#include "test_utils.h"

#include "test_knn.h"

CuSuite *UtilsGetSuite() {
    CuSuite *suite = CuSuiteNew();
    CuSuiteAddSuite(suite, KNNGetSuite());

    return suite;
}
