close all; clear variables; clc;

h = 0.5;

covarianceMatrix = cov(rand(1000, 3));
eigenValues = eig(covarianceMatrix);

S = h / (prod(eigenValues))