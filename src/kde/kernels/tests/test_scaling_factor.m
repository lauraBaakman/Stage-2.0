close all; clear variables; clc;

format long

% h = rand()
h = 0.757982987324507;

dimension = 3;
% dimension = round(rand() * 5);

covarianceMatrix = [0.089913920040088,   0.001355042095658,  -0.003748942313354;...
                    0.001355042095658,   0.079884621913451,  -0.001521957890412;...
                   -0.003748942313354,  -0.001521957890412,   0.086882503862659];

% covarianceMatrix = cov(rand(1000, dimension))
% covarianceMatrix = h * eye(dimension);
% covarianceMatrix = eye(dimension);
eigenValues = eig(covarianceMatrix);

% local_bandwidth = rand()
local_bandwidth = 0.905791937075619;

s = local_bandwidth * h * (prod(eigenValues)^(-1/dimension))
