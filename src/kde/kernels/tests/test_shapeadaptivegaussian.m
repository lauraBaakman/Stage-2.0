close all; clear variables; clc;

H = [2, -1, 0; -1, 2, -1; 0, -1, 2];

expectedResult = @(pattern, H) 1 / det(H) * mvnpdf(inv(H) * pattern);

x1 = [0.05; 0.05; 0.05];
x2 = [0.02; 0.03; 0.04];
x3 = [0.04; 0.05; 0.03];

expectedResult(x1, H) 
expectedResult(x2, H) 
expectedResult(x3, H)