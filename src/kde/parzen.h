//
// Created by Laura Baakman on 09/01/2017.
//

#ifndef PARZEN_H
#define PARZEN_H

#include "utils.h"
#include "kernels/kernels.h"

double parzen(double *pattern, Array *dataPoints, double windowWidth, double parzenFactor,
              KernelDensityFunction kernel, double kernelConstant);

#endif //PARZEN_H
