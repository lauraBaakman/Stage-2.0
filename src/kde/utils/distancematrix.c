#include "distancematrix.h"

void computeDistanceMatrix(Array *patterns, Array *distanceMatrix){

    double* a = patterns->data;
    double* b;
    double distance;

    for(int i = 0;
            i < patterns->length;
            i++, a+= patterns->stride)
    {
        b = a + patterns->stride;
        for(int j = i + 1;
                j < patterns->length;
                j++, b+= patterns->stride){
            printf("i = %d j = %d a[0] = %f b[0] %f\n", i, j, a[0], b[0]);

        }
    }
}