#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_matrix.h>
#include "knn.ih"

//Array* compute_k_nearest_neighbours(int k, int patternIdx, Array *patterns, Array *distanceMatrix,
//                                    Array *nearestNeighbours){
//    double* distances = arrayGetRowView(distanceMatrix, patternIdx);
//
//    ListElement* elements = toArrayOfListElements(distances, distanceMatrix->dimensionality);
//
//    listElementArraySort(elements, distanceMatrix->dimensionality);
//
//    getKNearestElements(elements, k, patterns, nearestNeighbours);
//
//    free(elements);
//
//    return nearestNeighbours;
//}

void compute_k_nearest_neighbours(int k, int patternIdx, gsl_matrix *patterns, gsl_matrix *distanceMatrix,
                                  gsl_matrix *outNearestNeighbours) {
    gsl_vector_view distances = gsl_matrix_row(distanceMatrix, patternIdx);

    size_t distanceCount = distanceMatrix->size1;

    ListElement* elements = toArrayOfListElements(&distances.vector);
    listElementArraySort(elements, distanceCount);

    getKNearestElements(elements, k,
                        patterns, outNearestNeighbours);

    free(elements);
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
}

void listElementArrayPrint(ListElement* elements, size_t numElements){
    for (size_t i = 0; i < numElements; ++i) {
        listElementPrint(&elements[i]);
    }
}

void listElementPrint(ListElement* element){
    printf("%f [%d] \n", *element->value, element->index);
}

void getKNearestElements(ListElement *sortedDistances, int k,
                         gsl_matrix *patterns, gsl_matrix *outNeighbours) {
    size_t idx = 0;
    gsl_vector_view pattern;
    for(size_t i = 0; i < k; i++){
        idx = sortedDistances[i].index;
        pattern = gsl_matrix_row(patterns, idx);
        gsl_matrix_set_row(outNeighbours, i, &pattern.vector);
    }
}

//void getKNearestElements(ListElement* sortedDistances, int k, Array* patterns, Array* neighbours){
//    int idx = 0;
//    double* pattern;
//    for (int i = 0; i < k; ++i) {
//        idx = sortedDistances[i].index;
//        pattern = arrayGetRowView(patterns, idx);
//        arraySetRow(neighbours, i, pattern);
//    }
//}
