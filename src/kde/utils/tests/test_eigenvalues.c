#include "test_eigenvalues.h"

#include "../../../lib/CuTestUtils.h"
#include "../../../test_utils.h"
#include "../gsl_utils.h"

#include "../eigenvalues.h"

void testEigenValues2D(CuTest *tc){
	double actual_matrix_data[] = {
		+0.081041292578536,  -0.003049670687501,
  		-0.003049670687501,  +0.083535264541089
	};
	gsl_matrix_view actual_matrix = gsl_matrix_view_array (actual_matrix_data, 2, 2);   		

	gsl_vector* actual_eigenValues = gsl_vector_alloc(2);
	gsl_matrix* actual_eigenVectors = gsl_matrix_alloc(2, 2);

	gsl_vector* expected_eigenValues = gsl_vector_alloc(2);
	gsl_vector_set(expected_eigenValues, 0, 0.078993515239071);
	gsl_vector_set(expected_eigenValues, 1, 0.085583041880554);

	computeEigenValues(&actual_matrix.matrix, actual_eigenValues, actual_eigenVectors);

	CuAssertVectorEquals(tc, expected_eigenValues, actual_eigenValues, delta);

	gsl_vector_free(actual_eigenValues);
	gsl_matrix_free(actual_eigenVectors);

	gsl_vector_free(expected_eigenValues);
}

void testEigenValues4D(CuTest *tc){
	double actual_matrix_data[] = {
  		+0.080710112672680, -0.004843823028562, -0.000075842450445, -0.001942952196866,
  		-0.004843823028562, +0.086617617552903, +0.000048285459557, -0.000200071313701,
  		-0.000075842450445, +0.000048285459557, +0.080620343074346, +0.000685495972908,
  		-0.001942952196866, -0.000200071313701, +0.000685495972908, +0.081253950324014
	};
	gsl_matrix_view actual_matrix = gsl_matrix_view_array (actual_matrix_data, 4, 4);   		

	gsl_vector* actual_eigenValues = gsl_vector_alloc(4);
	gsl_matrix* actual_eigenVectors = gsl_matrix_alloc(4, 4);

	gsl_vector* expected_eigenValues = gsl_vector_alloc(4);
	gsl_vector_set(expected_eigenValues, 0, 0.077173416611271);
	gsl_vector_set(expected_eigenValues, 1, 0.080384884829426);
	gsl_vector_set(expected_eigenValues, 2, 0.082227427958914);
	gsl_vector_set(expected_eigenValues, 3, 0.089416294224332);

	computeEigenValues(&actual_matrix.matrix, actual_eigenValues, actual_eigenVectors);
	// Sort for easier testing
	gsl_eigen_symmv_sort (actual_eigenValues, actual_eigenVectors, GSL_EIGEN_SORT_ABS_ASC);
	
	CuAssertVectorEquals(tc, expected_eigenValues, actual_eigenValues, delta);

	gsl_vector_free(actual_eigenValues);
	gsl_matrix_free(actual_eigenVectors);

	gsl_vector_free(expected_eigenValues);
}

void testEigenVectors(CuTest *tc){
	double actual_matrix_data[] = {
		+0.084495842251046, -0.002239528872347,
  		-0.002239528872347, +0.087218598293883
	};
	gsl_matrix_view actual_matrix = gsl_matrix_view_array (actual_matrix_data, 2, 2);   		

	gsl_vector* actual_eigenValues = gsl_vector_alloc(2);
	gsl_matrix* actual_eigenVectors = gsl_matrix_alloc(2, 2);

	double expected_eigenVectors_data[] = {
  		+0.871619750600529, -0.490182629601532,
  		+0.490182629601532, +0.871619750600529
	};
	gsl_matrix_view expected_eigenVectors = gsl_matrix_view_array (expected_eigenVectors_data, 2, 2);   		

	computeEigenValues(&actual_matrix.matrix, actual_eigenValues, actual_eigenVectors);
	// Sort for easier testing
	gsl_eigen_symmv_sort (actual_eigenValues, actual_eigenVectors, GSL_EIGEN_SORT_ABS_ASC);

	CuAssertMatrixEquals(tc, &expected_eigenVectors.matrix, actual_eigenVectors, delta);

	gsl_vector_free(actual_eigenValues);
	gsl_matrix_free(actual_eigenVectors);
}

void testInputMatrixIsReadOnly2D(CuTest *tc){
	double actual_matrix_data[] = {
		+0.084495842251046, -0.002239528872347,
  		-0.002239528872347, +0.087218598293883
	};
	gsl_matrix_view actual_matrix = gsl_matrix_view_array (actual_matrix_data, 2, 2);   		

	gsl_vector* actual_eigenValues = gsl_vector_alloc(2);
	gsl_matrix* actual_eigenVectors = gsl_matrix_alloc(2, 2);

	double expected_matrix_data[] = {
		+0.084495842251046, -0.002239528872347,
  		-0.002239528872347, +0.087218598293883
	};
	gsl_matrix_view expected_matrix = gsl_matrix_view_array (expected_matrix_data, 2, 2);   		

	computeEigenValues(&actual_matrix.matrix, actual_eigenValues, actual_eigenVectors);

	CuAssertMatrixEquals(tc, &expected_matrix.matrix, &actual_matrix.matrix, delta);

	gsl_vector_free(actual_eigenValues);
	gsl_matrix_free(actual_eigenVectors);
}

void testInputMatrixIsReadOnly4D(CuTest *tc){
	double actual_matrix_data[] = {
		+0.083084556753668,  -0.001861951087958,  +0.000791712690747,  -0.001077188220949,
		-0.001861951087958,  +0.083278979960787,  +0.002091045853742,  +0.000083057786867,
		+0.000791712690747,  +0.002091045853742,  +0.080849658550401,  -0.003187910001581,
		-0.001077188220949,  +0.000083057786867,  -0.003187910001581,  +0.083334427443704,
	};
	gsl_matrix_view actual_matrix = gsl_matrix_view_array (actual_matrix_data, 4, 4);   		

	gsl_vector* actual_eigenValues = gsl_vector_alloc(4);
	gsl_matrix* actual_eigenVectors = gsl_matrix_alloc(4, 4);

	double expected_matrix_data[] = {
		+0.083084556753668,  -0.001861951087958,  +0.000791712690747,  -0.001077188220949,
		-0.001861951087958,  +0.083278979960787,  +0.002091045853742,  +0.000083057786867,
		+0.000791712690747,  +0.002091045853742,  +0.080849658550401,  -0.003187910001581,
		-0.001077188220949,  +0.000083057786867,  -0.003187910001581,  +0.083334427443704,
	};
	gsl_matrix_view expected_matrix = gsl_matrix_view_array (expected_matrix_data, 4, 4);   		

	computeEigenValues(&actual_matrix.matrix, actual_eigenValues, actual_eigenVectors);

	CuAssertMatrixEquals(tc, &expected_matrix.matrix, &actual_matrix.matrix, delta);

	gsl_vector_free(actual_eigenValues);
	gsl_matrix_free(actual_eigenVectors);
}

CuSuite *EigenValuesGetSuite() {
    CuSuite *suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, testEigenValues2D);
    SUITE_ADD_TEST(suite, testEigenValues4D);
    SUITE_ADD_TEST(suite, testEigenVectors);
    SUITE_ADD_TEST(suite, testInputMatrixIsReadOnly2D);
    SUITE_ADD_TEST(suite, testInputMatrixIsReadOnly4D);
    return suite;
}
