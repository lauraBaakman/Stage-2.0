#include <gsl/gsl_matrix.h>

size_t numXs;
size_t numXis;
size_t dimension;


void readInput(int argc, char* argv[]){
	if(argc != 4){
		fprintf(stderr, "Expected usage:\n\t./benchmark.out numXs, numXis dimension\n");
		exit(-1);		
	}
	if (
		1 != sscanf(argv[1], "%zu", &numXs) ||
		1 != sscanf(argv[2], "%zu", &numXis) ||
		1 != sscanf(argv[3], "%zu", &dimension)
	){
		fprintf(stderr, "Sscanf failed.\n");
		exit(-1);
	}
	printf("numXs: %zu, numXis: %zu, dimension: %zu\n", numXs, numXis, dimension);	
}

int main(int argc, char* argv[]){
	readInput(argc, argv);

    gsl_matrix* xs = gsl_matrix_alloc(numXs, dimension);
    gsl_matrix_set_all(xs, 0.5);

    gsl_matrix_free(xs);
    return 0;
}