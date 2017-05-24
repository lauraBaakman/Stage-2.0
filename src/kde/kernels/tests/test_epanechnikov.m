close all; clear variables; clc;

kernel_variance = 16/ 21;

unitSphereVolume = @(d) (2 /d) * (pi^(d / 2) / gamma(d / 2));
epanechnikov = @(x) (1 - dot(x, x)) * (dot(x,x) < 1);
unitVarianceEpanechnikov = @(x) (1 / sqrt(kernel_variance))^length(x) * epanechnikov((1 / sqrt(kernel_variance) * x));

