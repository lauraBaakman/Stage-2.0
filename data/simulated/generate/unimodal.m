function [data, densities, Ns] = unimodal(N, d)
    %UNIMODAL generates an unimodal 3D dataset for testing purposes.
    
    %%INIT
    Ns = N;
    
    mean = repmat(50, 1, d);
    covariance = diag(repmat(10, 1 ,d));
    
    data = normalDistribution(mean, covariance, Ns);
    densities = mvnpdf(data, mean, covariance);