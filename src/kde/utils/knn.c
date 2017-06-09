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
    computeNearestNeighboursKD(&pattern.vector, (int) k);

    free(elements);
}

void computeNearestNeighboursKD(gsl_vector* pattern, int k){
    void* tree = kd_create(2);
    void* res;
    double data[] = {
            2, 3,
            5, 4,
            9, 6,
            4, 7,
            8, 1,
            7, 2
    };
    double pos[2] = {9, 2};
    kd_insert(tree, &data[0], NULL);
    kd_insert(tree, &data[2], NULL);
    kd_insert(tree, &data[4], NULL);
    kd_insert(tree, &data[6], NULL);
    kd_insert(tree, &data[8], NULL);
    kd_insert(tree, &data[10], NULL);

    res = kd_nearest_n(tree, pos, 3);
    while(!kd_res_end(res)) {
        kd_res_item(res, &data[0]);
        printf("%f %f %f\n", data[0], data[1], kd_res_dist(res));
        kd_res_next(res);
    }
    kd_res_free(res);
    kd_free(tree);
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

    buildKDTree(xs);
}

void nn_free(){
    gsl_matrix_free(g_distanceMatrix);
    kd_free(kdTree);
}

void buildKDTree(gsl_matrix* xs){
    kdTree = kd_create((int) xs->size2);

    gsl_vector_view row;

    double  x, y, z;

     for (size_t i = 0; i < xs->size1; i++)
     {
         row = gsl_matrix_row(xs, i);
         x = gsl_vector_get(&row.vector, 0);
         y = gsl_vector_get(&row.vector, 1);
         z = 0;

        kd_insert3(kdTree, x, y, z, NULL);
     }
}