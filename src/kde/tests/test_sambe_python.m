close all; clc; clear variables;

H1 = [+0.832940370216, -0.416470185108; -0.416470185108, +0.832940370216];
H4 = H1;

H2 = [0.832940370216, 0.416470185108; 0.416470185108, 0.832940370216];
H3 = H2;

H5 = [0.83294037, -0.41647019; -0.41647019,  0.83294037];

localBandwidths = [0.840896194314, 1.18920742746, 1.18920742746, 0.840896194314];
bandwidthmatrices(:,:,1) = H1;
bandwidthmatrices(:,:,2) = H2;
bandwidthmatrices(:,:,3) = H3;
bandwidthmatrices(:,:,4) = H4;
bandwidthmatrices(:,:,5) = H5;

x1 = [0; 0];
x2 = [0; 1];
x3 = [1; 0];
x4 = [1; 1];
x5 = [2; 2];

kernel = @(pattern, H, localBandwidth) 1 / det(localBandwidth * H) * mvnpdf(inv(localBandwidth * H) * pattern);

%% New Stuff
estimateDensity = @(x, Hs, lambdas) 1/4 * (...
    kernel(x - x1, Hs(:, :, 1), lambdas(1)) +...
    kernel(x - x2, Hs(:, :, 2), lambdas(2)) +...
    kernel(x - x3, Hs(:, :, 3), lambdas(3)) +...
    kernel(x - x4, Hs(:, :, 4), lambdas(4)) ...
);

densityx1 = estimateDensity(x1, bandwidthmatrices, localBandwidths)
densityx2 = estimateDensity(x2, bandwidthmatrices, localBandwidths)
% densityx3 = estimateDensity(x3, bandwidthmatrices, localBandwidths)
densityx4 = estimateDensity(x4, bandwidthmatrices, localBandwidths)
densityx5 = estimateDensity(x5, bandwidthmatrices, localBandwidths)

%% Old stuff

estimateDensityOld = @(x, H, bandwidths) 1/4 * (...
    kernel(x - x1, H, bandwidths(1)) +...
    kernel(x - x2, H, bandwidths(2)) +...
    kernel(x - x3, H, bandwidths(3)) +...
    kernel(x - x4, H, bandwidths(4)) ...
);

olddensityx1 = estimateDensityOld(x1, H1, localBandwidths)
olddensityx4 = estimateDensityOld(x4, H4, localBandwidths)
olddensityx5 = estimateDensityOld(x5, H5, localBandwidths)