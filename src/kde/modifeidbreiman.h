//
// Created by Laura Baakman on 26/01/2017.
//

#ifndef KERNELS_MODIFEID_BREIMAN_H
#define KERNELS_MODIFEID_BREIMAN_H

#include "utils.h"

double mbe_epanechnikov(double *pattern, Array *dataPoints,
                        double globalBandwidth, Array *localBandwidths,
                        double epanechnikovFactor, double parzenFactor);

#endif //KERNELS_MODIFEID_BREIMAN_H
