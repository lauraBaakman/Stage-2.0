//
// Created by Laura Baakman on 26/01/2017.
//

#ifndef KERNELS_MODIFEID_BREIMAN_H
#define KERNELS_MODIFEID_BREIMAN_H

#include "utils.h"
#include "kernels/kernels.h"

double modifiedBreimanFinalDensity(double *pattern, Array *dataPoints,
                                   double globalBandwidth, Array *localBandwidths,
                                   double kernelConstant, KernelDensityFunction kernel);

#endif //KERNELS_MODIFEID_BREIMAN_H
