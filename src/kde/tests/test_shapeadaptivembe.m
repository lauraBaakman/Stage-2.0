close all; clear variables; clc;

xis = [0, 0; 0, 1; 1, 0; 1, 1];

xs = [0, 0; 1, 1];

[N, d] = size(xis);

h = 1 / log(4);

localBandwidths = [0.840896194313949, 1.189207427458816, 1.189207427458816, 0.840896194313949];

k = 3;

%% estimate the density of one pattern
x = xs(1, :);

ck = [0, 0; 0, 1; 1, 0];

covarianceMatrix = cov(ck, 1);

eigenValues = eig(covarianceMatrix);

S = h * (prod(eigenValues)^(-1/d));

H(:, :, 1) = localBandwidths(1) * S * covarianceMatrix;
H(:, :, 2) = localBandwidths(2) * S * covarianceMatrix;
H(:, :, 3) = localBandwidths(3) * S * covarianceMatrix;
H(:, :, 4) = localBandwidths(4) * S * covarianceMatrix;

fhatTerm = @(H, x, xi) 1 / det(H) * mvnpdf((xi - x) * inv(H));

fhatTerms = [fhatTerm(H(:, :, 1), x, xis(1)),...
    fhatTerm(H(:, :, 2), x, xis(2)),...
    fhatTerm(H(:, :, 3), x, xis(3)),...
    fhatTerm(H(:, :, 4), x, xis(4))];

fhat = 1 / N * sum(fhatTerms)