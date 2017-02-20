#include "knn.ih"

Array* compute_k_nearest_neighbours(int k, int patternIdx, Array *patterns, Array *distanceMatrix,
                                    Array *nearestNeighbours){
    double* distances = arrayGetRow(distanceMatrix, patternIdx);

    ListElement* elements = toArrayOfListElements(distances, distanceMatrix->dimensionality);

    listElementArraySort(elements, distanceMatrix->dimensionality);

    getKNearestElements(elements, k, patterns, nearestNeighbours);

    free(elements);

    return nearestNeighbours;
}

ListElement* toArrayOfListElements(double *values, int numValues){
    ListElement* elements = malloc(numValues * sizeof(ListElement));
    for (int i = 0; i < numValues; ++i) {
        elements[i].index = i;
        elements[i].value = &values[i];
    }
    return elements;
}

void listElementArraySort(ListElement* elements, int numElements){
    qsort((void *) elements, (size_t) numElements, sizeof(ListElement), listElementCompare);
}

int listElementCompare(const void *s1, const void *s2){
    struct ListElement *e1 = (struct ListElement *)s1;
    struct ListElement *e2 = (struct ListElement *)s2;

    if (*(e1->value) < *(e2->value)) return -1;
    if (*(e1->value) == *(e2->value)) return 0;
    if (*(e1->value) > *(e2->value)) return +1;
}

void listElementArrayPrint(ListElement* elements, int numElements){
    for (int i = 0; i < numElements; ++i) {
        listElementPrint(&elements[i]);
    }
}

void listElementPrint(ListElement* element){
    printf("%f [%d] \n", *element->value, element->index);
}

void getKNearestElements(ListElement* sortedDistances, int k, Array* patterns, Array* neighbours){
    int idx = 0;
    double* pattern;
    for (int i = 0; i < k; ++i) {
        idx = sortedDistances[i].index;
        pattern = arrayGetRow(patterns, idx);
        arraySetRow(neighbours, i, pattern);
    }
}