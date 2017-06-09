#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_matrix.h>
#include "knn.ih"

static gsl_matrix* g_distanceMatrix;
struct kdtree* kdTree;

void computeKNearestNeighbours(size_t k, size_t patternIdx, gsl_matrix *patterns, gsl_matrix *outNearestNeighbours) {
    gsl_vector_view distances = gsl_matrix_row(g_distanceMatrix, patternIdx);

    size_t distanceCount = g_distanceMatrix->size1;

    ListElement* elements = toArrayOfListElements(&distances.vector);
    listElementArraySort(elements, distanceCount);

    getKNearestElements(elements, k,
                        patterns, outNearestNeighbours);

    gsl_vector_view pattern = gsl_matrix_row(patterns, patternIdx);
    computeNearestNeighboursKDHelper(&pattern.vector, (int) k);

    free(elements);
}

void computeNearestNeighboursKDHelper(gsl_vector *pattern, int k){
    gsl_matrix* xs = gsl_matrix_alloc(6, 3);
    gsl_matrix_set(xs, 0, 0, 2); gsl_matrix_set(xs, 0, 1, 3);
    gsl_matrix_set(xs, 1, 0, 5); gsl_matrix_set(xs, 1, 1, 4);
    gsl_matrix_set(xs, 2, 0, 9); gsl_matrix_set(xs, 2, 1, 6);
    gsl_matrix_set(xs, 3, 0, 4); gsl_matrix_set(xs, 3, 1, 7);
    gsl_matrix_set(xs, 4, 0, 8); gsl_matrix_set(xs, 4, 1, 1);
    gsl_matrix_set(xs, 5, 0, 7); gsl_matrix_set(xs, 5, 1, 2);

    // Allocate memory
    k = 3;
    struct kdtree* tree = kd_create((int) xs->size2);
    gsl_matrix* result = gsl_matrix_alloc(k, (size_t) kd_dimension(tree));

    // Build Tree
    buildKDTree(tree, xs);

    // KNN
    gsl_vector* position = gsl_vector_alloc(kd_dimension(tree));
    gsl_vector_set(position, 0, 9);
    gsl_vector_set(position, 1, 2);
    gsl_vector_set(position, 2, 0);

    computeNearestNeighboursKD(tree, position, k, result);

    printf("Nearest neighbours\n");
    gsl_matrix_print(stdout, result);

    // Free Memory
    kd_free(tree);
    gsl_matrix_free(xs);
    gsl_matrix_free(result);
    gsl_vector_free(position);
}

void computeNearestNeighboursKD(struct kdtree* tree, gsl_vector* pattern, int k, gsl_matrix* neighbours){
    double* resultRow;
    struct kdres* res = kd_nearest_n(tree, pattern->data, k);
    for(int i = 0; i < kd_res_size(res); i++, kd_res_next(res)){
        resultRow = &neighbours->data[i * neighbours->tda];
        kd_res_item(res, resultRow);
    }
    kd_res_free(res);
}

ListElement* toArrayOfListElements(gsl_vector *distances){
    ListElement* elements = malloc(distances->size * sizeof(ListElement));
    for (size_t i = 0; i < distances->size; ++i) {
        elements[i].index = i;
        elements[i].value = &(distances->data[i]);
    }
    return elements;
}

void listElementArraySort(ListElement* elements, size_t numElements){
    qsort((void *) elements, numElements, sizeof(ListElement), listElementCompare);
}

int listElementCompare(const void *s1, const void *s2){
    struct ListElement *e1 = (struct ListElement *)s1;
    struct ListElement *e2 = (struct ListElement *)s2;

    if (*(e1->value) < *(e2->value)) return -1;
    if (*(e1->value) == *(e2->value)) return 0;
    if (*(e1->value) > *(e2->value)) return +1;
    return 0; //avoid compile warnings
}

void listElementArrayPrint(ListElement* elements, size_t numElements){
    for (size_t i = 0; i < numElements; ++i) {
        listElementPrint(&elements[i]);
    }
}

void listElementPrint(ListElement* element){
    printf("%f [%lu] \n", *element->value, element->index);
}

void getKNearestElements(ListElement *sortedDistances, size_t k,
                         gsl_matrix *patterns, gsl_matrix *outNeighbours) {
    size_t idx = 0;
    gsl_vector_view pattern;
    for(size_t i = 0; i < (size_t) k; i++){
        idx = sortedDistances[i].index;
        pattern = gsl_matrix_row(patterns, idx);
        gsl_matrix_set_row(outNeighbours, i, &pattern.vector);
    }
}

void computeDistanceMatrix(gsl_matrix *patterns, gsl_matrix *distanceMatrix) {
    gsl_matrix_set_zero(distanceMatrix);

    gsl_vector_view a, b;

    size_t patternCount = patterns->size1;

    double distance;
    for(size_t i = 0; i < patternCount; i++){
        a = gsl_matrix_row(patterns, i);
        for(size_t j = i + 1; j < patternCount; j++){
            b = gsl_matrix_row(patterns, j);

            distance = squaredEuclidean(&a.vector, &b.vector);

            gsl_matrix_set(distanceMatrix, i, j, distance);
            gsl_matrix_set(distanceMatrix, j, i, distance);
        }
    }
}

double squaredEuclidean(gsl_vector* a, gsl_vector* b){
    size_t vectorDimension = a->size;
    double distance = 0;
    double difference;
    for(size_t i = 0; i < vectorDimension; i++){
        difference = a->data[i] - b->data[i];
        distance += difference * difference;
    }
    return distance;
}

void nn_prepare(gsl_matrix* xs){
    g_distanceMatrix = gsl_matrix_alloc(xs->size1, xs->size1);
    computeDistanceMatrix(xs, g_distanceMatrix);
}

void nn_free(){
    gsl_matrix_free(g_distanceMatrix);
    kd_free(kdTree);
}

void buildKDTree(struct kdtree* tree, gsl_matrix* xs){
    double *row;
    for (size_t i = 0; i < xs->size1; ++i) {
        row = &xs->data[i * xs->tda];
        kd_insert(tree, row, NULL);
    }
}