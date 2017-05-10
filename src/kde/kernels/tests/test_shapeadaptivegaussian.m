close all; clear variables; clc;

H = [2, -1, 0; -1, 2, -1; 0, -1, 2];

expectedResult = @(pattern, H, localBandwidth) 1 / det(localBandwidth * H) * mvnpdf(inv(localBandwidth * H) * pattern);

x1 = [0.05; 0.05; 0.05];
x2 = [0.02; 0.03; 0.04];
x3 = [0.04; 0.05; 0.03];

lambda1 = 0.5;
lambda2 = 0.7;
lambda3 = 0.2;

expectedResult(x1, H, 1); 
expectedResult(x2, H, 1); 
expectedResult(x3, H, 1);


expectedResult(x1, H, lambda1) 
expectedResult(x2, H, lambda2)
expectedResult(x3, H, lambda3)