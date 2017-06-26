H1 = [+0.832940370216, -0.416470185108; -0.416470185108, +0.832940370216];
H4 = H1;

H2 = [0.832940370216, 0.416470185108; 0.416470185108, 0.832940370216];
H3 = H2;

bandwidths = [0.840896194314, 1.18920742746, 1.18920742746, 0.840896194314];

x1 = [0; 0];
x2 = [0; 1];
x3 = [1; 0];
x4 = [1; 1];

kernel = @(pattern, H, localBandwidth) 1 / det(localBandwidth * H) * mvnpdf(inv(localBandwidth * H) * pattern);
estimateDensity = @(x, H, bandwidths) 1/4 * (...
    kernel(x - x1, H, bandwidths(1)) +...
    kernel(x - x2, H, bandwidths(2)) +...
    kernel(x - x3, H, bandwidths(3)) +...
    kernel(x - x4, H, bandwidths(4))  ...
);

densityx1 = estimateDensity(x1, H1, bandwidths)
densityx2 = estimateDensity(x2, H2, bandwidths)
densityx3 = estimateDensity(x3, H3, bandwidths)
densityx4 = estimateDensity(x4, H4, bandwidths)
