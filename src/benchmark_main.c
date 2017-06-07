#include <gsl/gsl_matrix.h>

size_t numXs = 60;
size_t d = 3;


int main(){
	printf("Hoi!\n");

    gsl_matrix* xs = gsl_matrix_alloc(numXs, d);
    gsl_matrix_set_all(xs, 0.5);

    gsl_matrix_free(xs);
    return 0;
}