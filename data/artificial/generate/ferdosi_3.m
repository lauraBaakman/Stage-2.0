function [ data, densities, Ns] = ferdosi_3( N, d)
    %FERDOSI_3 Dataset one from ferdosi et al.
    % [DATA, DENSITIES, RATIOS] = FERDOSI_2(N, D) generates a D dimensional
    % dataset with N patterns. Where DATA is a N x D matrix where each row
    % is a pattern, DENSITIES is is a N x 1 vector with the true
    % densities of the patterns.
    
    %% INIT
    ratios = [1/6, 1/6, 1/6, 1/6, 1/3];
    
    Ns = ceil(N * ratios);
    clear rows;
    [startRow, endRow] = rows(Ns);    
    %% The Gaussian 1 data
    means(:,:,1) = [24, 10, 10];
    covariances(:,:,1) = diag(repmat(2, 1 ,d));
    
    data(startRow:endRow, :)= normalDistribution(means(:,:,1), covariances(:,:,1), Ns(1));
    [startRow, endRow] = rows(Ns);
    %% The Gaussian 2 data
    means(:,:,2) = [33, 70, 40];
    covariances(:,:,2) = diag(repmat(10, 1 ,d));
    
    data(startRow:endRow, :)= normalDistribution(means(:,:,2), covariances(:,:,2), Ns(2));
    [startRow, endRow] = rows(Ns);    
    %% The Gaussian 3 data
    means(:,:,3) = [90, 20, 80];
    covariances(:,:,3) = diag(repmat(5, 1 ,d));
    
    data(startRow:endRow, :)= normalDistribution(means(:,:,3), covariances(:,:,3), Ns(3));    
    [startRow, endRow] = rows(Ns);        
    %% The Gaussian 4 data
    means(:,:,4) = [60, 80, 23];
    covariances(:,:,4) = diag(repmat(5, 1 ,d));
    
    data(startRow:endRow, :)= normalDistribution(means(:,:,4), covariances(:,:,4), Ns(4));    
    
    %% The uniform data
    lower = 0;
    upper = 100;
    
    [data(end + 1:end + Ns(5), :), uniformDensity] = uniformDistribution(lower, upper, Ns(5), d);    
    
    %% Combine data
    densities = ...
        ratios(1) .* mvnpdf(data, means(:,:,1), covariances(:,:,1)) + ...
        ratios(2) .* mvnpdf(data, means(:,:,2), covariances(:,:,2)) + ...
        ratios(3) .* mvnpdf(data, means(:,:,3), covariances(:,:,3)) + ...        
        ratios(4) .* mvnpdf(data, means(:,:,4), covariances(:,:,4)) + ...                
        ratios(5) .* (ones(sum(Ns),1) * uniformDensity);
    
end

