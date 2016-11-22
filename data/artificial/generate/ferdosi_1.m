function [ data, densities, Ns] = ferdosi_1( N, d)
    %FERDOSI_1 Dataset one from ferdosi et al.
    % [DATA, DENSITIES, RATIOS] = FERDOSI_1(N, D) generates a D dimensional
    % dataset with N patterns. Where DATA is a N x D matrix where each row
    % is a pattern, DENSITIES is is a N x 1 vector with the true
    % densities of the patterns .
    
    %% INIT
    ratios = [2/3, 1/3];
    Ns = ceil(N * ratios);
    
    %% The Gaussian data
    mean = repmat(50, 1, d);
    covariance = diag(repmat(30, 1 ,d));
    
    gaussianData= normalDistribution(mean, covariance, Ns(1));
    
    %% The uniform data
    lower = 0;
    upper = 100;
    
    [uniformData, uniformDensity] = uniformDistribution(lower, upper, Ns(2), d);
    
    %% Combine data
    data = [gaussianData; uniformData];
    densities = ...
        ratios(1) .* mvnpdf(data, mean, covariance) + ...
        ratios(2) .* (ones(sum(Ns),1) * uniformDensity);
    
end

