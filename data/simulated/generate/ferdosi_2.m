function [ data, densities, Ns] = ferdosi_2( N, d)
    %FERDOSI_2 Dataset one from ferdosi et al.
    % [DATA, DENSITIES, RATIOS] = FERDOSI_2(N, D) generates a D dimensional
    % dataset with N patterns. Where DATA is a N x D matrix where each row
    % is a pattern, DENSITIES is is a N x 1 vector with the true
    % densities of the patterns.
    
    %% INIT
    ratios = [1/3, 1/3, 1/3];
    Ns = ceil(N * ratios);
    
    %% The Gaussian 1 data
    mean1 = repmat(25, 1, d);
    covariance1 = diag(repmat(5, 1 ,d));
    
    gaussian1Data = normalDistribution(mean1, covariance1, Ns(1));
    
    %% The Gaussian 2 data
    mean2 = repmat(65, 1, d);
    covariance2 = diag(repmat(20, 1 ,d));
    
    gaussian2Data = normalDistribution(mean2, covariance2, Ns(2));
    
    %% The uniform data
    lower = 0;
    upper = 100;
    
    [uniformData, uniformDensity] = uniformDistribution(lower, upper, Ns(3), d);    
    
    %% Combine data
    data = [gaussian1Data; gaussian2Data; uniformData];
    densities = ...
        ratios(1) .* mvnpdf(data, mean1, covariance1) + ...
        ratios(2) .* mvnpdf(data, mean2, covariance2) + ...
        ratios(3) .* (ones(sum(Ns),1) * uniformDensity);
    
end

