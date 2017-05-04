close all; clear variables; clc;

H = [4, 7; 2, 6];

expectedResult = @(pattern, H, localBandwidth) 1 / det(localBandwidth * H) * mvnpdf(inv(localBandwidth * H) * pattern);

x1 = [0.05; 0.05];
x2 = [0.02; 0.03];
x3 = [0.04; 0.05];

expectedResult(x1, H, 1); 
expectedResult(x2, H, 1); 
expectedResult(x3, H, 1);


expectedResult(x1, H, 0.5) 
expectedResult(x2, H, 0.5)
expectedResult(x3, H, 0.5)