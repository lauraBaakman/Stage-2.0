close all; clear variables; clc;

kernel_variance_factor = 1 / sqrt(1/5);

unitSphereVolume = @(d) (2 /d) * (pi^(d / 2) / gamma(d / 2));
epanechnikov = @(x) ((length(x) + 2) / (2 * unitSphereVolume(length(x)))) * (1 - dot(x, x)) * (dot(x,x) < 1);
expectedResult = @(pattern, H) 1 / det(kernel_variance_factor * H) ...
    * epanechnikov(inv(kernel_variance_factor * H) * pattern);

%% 3D
H = [2, -1, 0; -1, 2, -1; 0, -1, 2];

x1 = [0.05; 0.05; 0.05];
x2 = [0.02; 0.03; 0.04];
x3 = [0.04; 0.05; 0.03];

expectedResult(x1, H)
expectedResult(x2, H) 
expectedResult(x3, H)