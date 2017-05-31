close all; clear variables; clc;

kernel_variance = 16/ 21;

unitSphereVolume = @(d) (2 /d) * (pi^(d / 2) / gamma(d / 2));
epanechnikov = @(x) ((length(x) + 2) / (2 * unitSphereVolume(length(x)))) * (1 - dot(x, x)) * (dot(x,x) < 1);
expectedResult = @(pattern, H, localBandwidth) 1 / det(sqrt(kernel_variance) * localBandwidth * H) ...
    * epanechnikov(inv(sqrt(kernel_variance) * localBandwidth * H) * pattern);

%% 3D

H = [2, -1, 0; -1, 2, -1; 0, -1, 2];

x1 = [0.05; 0.05; 0.05];
x2 = [0.02; 0.03; 0.04];
x3 = [0.04; 0.05; 0.03];

lambda1 = 0.5;
lambda2 = 0.7;
lambda3 = 0.2;

expectedResult(x1, H, 1)
expectedResult(x2, H, 1) 
expectedResult(x3, H, 1)

expectedResult(x1, H, lambda1) 
expectedResult(x2, H, lambda2)
expectedResult(x3, H, lambda3)