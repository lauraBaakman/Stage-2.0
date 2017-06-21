close all; clear variables; clc;

kernel_variance_factor = 1 / sqrt(1 / 5);

unitSphereVolume = @(d) (2 /d) * (pi^(d / 2) / gamma(d / 2));
epanechnikov = @(x) ((length(x) + 2) / (2 * unitSphereVolume(length(x)))) * (1 - dot(x, x)) * (dot(x,x) < 1);
unitVarianceEpanechnikov = @(x) sqrt(5)^(-length(x)) * ((length(x) + 2) / (2 * unitSphereVolume(length(x)))) * (1 - 1/5 * dot(x, x)) * (dot(x,x) < sqrt(5));

%% 1D
sprintf('1D\n')
unitVarianceEpanechnikov(5)
unitVarianceEpanechnikov(-0.5)
unitVarianceEpanechnikov(0.5)

%% 2D
sprintf('2D\n')
unitVarianceEpanechnikov([5, 0.5])
unitVarianceEpanechnikov([0.5, 0.5])
unitVarianceEpanechnikov([0.3, 0.5])

%% 3D
sprintf('3D\n')
unitVarianceEpanechnikov([0.5, 0.5, 0.5])

%% 4D
sprintf('4D\n')
unitVarianceEpanechnikov([0.3, 0.5, 0.5, 0.5])
unitVarianceEpanechnikov([0.2, 0.2, 0.1, 0.5])

%% 5D
sprintf('5D\n')
unitVarianceEpanechnikov([0.3, 0.5, 0.5, 0.5, 0.4])
unitVarianceEpanechnikov([0.2, 0.2, 0.1, 0.5, 0.01])