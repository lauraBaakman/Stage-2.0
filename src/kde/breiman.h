//
// Created by Laura Baakman on 26/01/2017.
//

#ifndef KERNELS_MBE_H
#define KERNELS_MBE_H

#include "utils.h"

double mbe_epanechnikov(double *pattern, Array *dataPoints, double windowWidth, double parzenFactor,
                           double epanechnikovFactor);

#endif //KERNELS_MBE_H
