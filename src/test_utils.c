#include "test_utils.h"

#include "omp.h"

static int g_oldNumThreads;

void limit_num_threads_to(int numThreads){
    #pragma omp parallel 
    { 
        g_oldNumThreads = omp_get_num_threads(); 
    }    
    omp_set_dynamic(0); 
    omp_set_num_threads(numThreads); 
}

void reset_omp(){
    omp_set_dynamic(1); 
    omp_set_num_threads(g_oldNumThreads); 
}