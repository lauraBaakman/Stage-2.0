close all; clear variables; clc;

kernel_variance = 16/ 21;

unitSphereVolume = @(d) (2 /d) * (pi^(d / 2) / gamma(d / 2));
epanechnikov = @(x) ((length(x) + 2) / (2 * unitSphereVolume(length(x)))) * (1 - dot(x, x)) * (dot(x,x) < 1);
epanechnikovKernel = @(x) (1 / sqrt(kernel_variance))^length(x) * epanechnikov((1 / sqrt(kernel_variance) * x));

mbe_singlePattern = @(x, xs, h, kernel) (h)^(-length(x)) * kernel((x - xs) ./ h);
mbe = @(x, xs, general_bandwidth, kernel) 1 / 3 * (...
    mbe_singlePattern(x, xs(1, :), general_bandwidth, kernel) + ...
    mbe_singlePattern(x, xs(2, :), general_bandwidth, kernel) + ...
    mbe_singlePattern(x, xs(3, :), general_bandwidth, kernel));

%% Shared Variables

xi_s = [-1, -1; 0, 0; 0.5, 0.5];
x_s = [0, 0; 0.25, 0.5];
general_bandwidth = 4;

%% Epanechnikov
result_epanechnikov_1 = mbe(x_s(1, :), xi_s, general_bandwidth, epanechnikovKernel)
result_epanechnikov_2 = mbe(x_s(2, :), xi_s, general_bandwidth, epanechnikovKernel)