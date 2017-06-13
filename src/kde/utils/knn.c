
#include "knn.ih"

struct kdtree* g_tree;

void computeKNearestNeighbours(gsl_vector *pattern, size_t k, gsl_matrix *neighbours) {
    double* resultRow;
    struct kdres* res = kd_nearest_n(g_tree, pattern->data, (int) k);
    for(int i = 0; i < kd_res_size(res); i++, kd_res_next(res)){
        resultRow = &neighbours->data[i * neighbours->tda];
        kd_res_item(res, resultRow);
    }
    kd_res_free(res);
}

void nn_prepare(gsl_matrix* xs){
    g_tree = kd_create((int) xs->size2);
    buildKDTree(xs);
}

void nn_free(){
    kd_free(g_tree);
}

void buildKDTree(gsl_matrix* xs){
    double *row;
    for (size_t i = 0; i < xs->size1; ++i) {
        row = &xs->data[i * xs->tda];
        kd_insert(g_tree, row, NULL);
    }
}