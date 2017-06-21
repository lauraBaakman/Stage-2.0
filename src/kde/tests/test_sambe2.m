close all; clear variables; clc;

xis = [0, 0; 0, 1; 1, 0; 1, 1];
xis = xis';

xs = [0, 0; 1, 1];
xs = xs';

[d, N] = size(xis);

h = 1 / log(4);

localBandwidths = [0.840896194313949, 1.189207427458816, 1.189207427458816, 0.840896194313949];

k = 3;

%% estimate the density of one pattern Gaussian Kernel
x = xs(:, 2);

ck = [0, 0; 0, 1; 1, 0];

covarianceMatrix = cov(ck, 1);

eigenValues = eig(covarianceMatrix);

S = h * (prod(eigenValues)^(-1/d));

H(:, :, 1) = localBandwidths(1) * S * covarianceMatrix;
H(:, :, 2) = localBandwidths(2) * S * covarianceMatrix;
H(:, :, 3) = localBandwidths(3) * S * covarianceMatrix;
H(:, :, 4) = localBandwidths(4) * S * covarianceMatrix;

fhatTerm = @(H, c, ci) 1 / det(H) * mvnpdf(inv(H) * (c - ci));


fhatTerms = [fhatTerm(H(:, :, 1), x, xis(:, 1)),...
    fhatTerm(H(:, :, 2), x, xis(:, 2)),...
    fhatTerm(H(:, :, 3), x, xis(:, 3)),...
    fhatTerm(H(:, :, 4), x, xis(:, 4))];

fhat = 1 / N * sum(fhatTerms);

%% estimate the density of one pattern Epanechnikov Kernel
xis = [0, 0; 0, 1; 1, 0; 1, 1]';

ck = [0, 0; 0, 1; 1, 0];

covarianceMatrix = cov(ck, 1);

eigenValues = eig(covarianceMatrix);

S = h * (prod(eigenValues)^(-1/d));

H(:, :, 1) = localBandwidths(1) * S * covarianceMatrix;
H(:, :, 2) = localBandwidths(2) * S * covarianceMatrix;
H(:, :, 3) = localBandwidths(3) * S * covarianceMatrix;
H(:, :, 4) = localBandwidths(4) * S * covarianceMatrix;

unitSphereVolume = @(d) (2 /d) * (pi^(d / 2) / gamma(d / 2));
unitSphereConstant = @(d) ((d + 2) / (2 * unitSphereVolume(d)));
epanechnikov = @(x) sqrt(5)^(-length(x)) * unitSphereConstant(length(x))  * (1 - 1/5 * dot(x, x)) * (dot(x,x) < sqrt(5));
fhatTerm = @(H, c, ci) 1 / det(H) * epanechnikov(inv(H) * (c - ci));

fhat = @(x) 1/ N * (...
    fhatTerm(H(:, :, 1), x, xis(:, 1)) + ...
    fhatTerm(H(:, :, 2), x, xis(:, 2)) + ...
    fhatTerm(H(:, :, 3), x, xis(:, 3)) + ...
    fhatTerm(H(:, :, 4), x, xis(:, 4)) ...
);

fhat(xis(:,1))
fhat(xis(:,2))
fhat(xis(:,3))
fhat(xis(:,4))