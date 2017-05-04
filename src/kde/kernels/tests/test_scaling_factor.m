close all; clear variables; clc;

format long

h = rand()
dimension = round(rand() * 5);

covarianceMatrix = cov(rand(1000, dimension))
% covarianceMatrix = h * eye(dimension);
% covarianceMatrix = eye(dimension);
eigenValues = eig(covarianceMatrix);

s = h * (prod(eigenValues)^(-1/dimension))
