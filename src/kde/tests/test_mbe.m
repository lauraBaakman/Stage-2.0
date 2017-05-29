close all; clear variables; clc;

kernel_variance = 16/ 21;

unitSphereVolume = @(d) (2 /d) * (pi^(d / 2) / gamma(d / 2));
epanechnikov = @(x) ((length(x) + 2) / (2 * unitSphereVolume(length(x)))) * (1 - dot(x, x)) * (dot(x,x) < 1);
epanechnikovKernel = @(x) (1 / sqrt(kernel_variance))^length(x) * epanechnikov((1 / sqrt(kernel_variance) * x));

mbe_singlePattern = @(x, xs, h, l, kernel) (l * h)^(-length(x)) * kernel((x - xs) ./ (l * h));
mbe = @(x, xs, general_bandwidth, local_bandwidths, kernel) 1 / 3 * (...
    mbe_singlePattern(x, xs(1, :), general_bandwidth, local_bandwidths(1), kernel) + ...
    mbe_singlePattern(x, xs(2, :), general_bandwidth, local_bandwidths(2), kernel) + ...
    mbe_singlePattern(x, xs(3, :), general_bandwidth, local_bandwidths(3), kernel));

%% Shared Variables

xi_s = [-1, -1; 1, 1; 0, 0];
x_s = [0, 0; 1, 1; 0, 1];
local_bandwidths = [10, 20, 50];
general_bandwidth = 0.5;

result_epanechnikov_1 = mbe(x_s(1, :), xi_s, general_bandwidth, local_bandwidths, epanechnikovKernel)