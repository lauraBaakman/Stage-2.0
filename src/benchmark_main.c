#include <time.h>

#include <gsl/gsl_matrix.h>

#include "kde/parzen.h"
#include "kde/kernels/kernels.h"

size_t numXs;
size_t numXis;
size_t dimension;


void readInput(int argc, char* argv[]){
	if(argc != 4){
		fprintf(stderr, "Expected usage:\n\t./benchmark.out numXs, numXis dimension\n");
		exit(-1);		
	}
	if (
			1 != sscanf(argv[1], "%zu", &numXs)
		|| 	1 != sscanf(argv[2], "%zu", &numXis)
		||	1 != sscanf(argv[3], "%zu", &dimension)
	){
		fprintf(stderr, "Sscanf failed.\n");
		exit(-1);
	}
	printf("numXs: %zu, numXis: %zu, dimension: %zu\n", numXs, numXis, dimension);	
}

clock_t benchmarkParzen(){
    gsl_matrix* xs = gsl_matrix_alloc(numXs, dimension);
    gsl_matrix_set_all(xs, 0.5);

    gsl_matrix* xis = gsl_matrix_alloc(numXis, dimension);
    gsl_matrix_set_all(xis, 0.5);

    gsl_vector* densities = gsl_vector_alloc(numXs);

    double windowWidth = 4;

    SymmetricKernel kernel = selectSymmetricKernel(EPANECHNIKOV);

    clock_t tic = clock();
    parzen(xs, xis, windowWidth, kernel, densities);
    clock_t toc = clock();

    gsl_vector_free(densities);
    gsl_matrix_free(xis);
    gsl_matrix_free(xs);

    return toc - tic;	
}

void printTimeInformation(clock_t tictoc, char* estimator){
	 printf("%s: %f seconds\n", estimator, (double)(tictoc) / CLOCKS_PER_SEC);
}

int main(int argc, char* argv[]){
	readInput(argc, argv);

	printTimeInformation(
		benchmarkParzen(), "parzen"
	);

    return 0;
}