function [ data, densities, Ns] = ferdosi_4( N, d)
    %FERDOSI_4 Dataset one from ferdosi et al.
    % [DATA, DENSITIES, RATIOS] = FERDOSI_2(N, D) generates a D dimensional
    % dataset with N patterns. Where DATA is a N x D matrix where each row
    % is a pattern, DENSITIES is is a N x 1 vector with the true
    % densities of the patterns.
    
    %% INIT
    ratios = [1/2, 1/2];
    
    Ns = ceil(N * ratios);
    
    clear rows;
    [startRow, endRow] = rows(Ns);    
    
    %% The wall data
    lower1 = 0;
    upper1 = 100;

    data(startRow:endRow, 1:2)= blockUniformDistribution([lower1, lower1], [upper1, upper1], Ns(1), 2);
    
    mean1 = 50;
    covariance1 = 5;
    data(startRow:endRow, 3)=  normalDistribution(mean1, covariance1, Ns(1));
    
    [startRow, endRow] = rows(Ns);
    %% The filament 2 data
    mean2 = repmat(50, 1, 2);
    covariance2 = diag(repmat(5, 1 ,2));
    
    data(startRow:endRow, 1:2)= normalDistribution(mean2, covariance2, Ns(2));
    
    
    lower2 = 0;
    upper2 = 100;
    data(startRow:endRow, 3) = blockUniformDistribution(lower2, upper2, Ns(2), 1);
    
    %% Compute densities
    
    %% Prepare density computation for wall
    wallDensity = computeWallDensity(lower1, upper1);
    wallSpace =  computeWallSpace(data(:,1:2), [lower1, lower1], [upper1, upper1]);
    
    
    %% Prepare density compution for filament 
    filamentDensity = computeWallDensity(lower2, upper2);
    filamentSpace = computeWallSpace(data(:,3), lower2, upper2);
    
    densities = ratios(1) * wallDensity .* wallSpace .* mvnpdf(data(:, 3), mean1, covariance1) + ...
                ratios(2) * filamentDensity .* filamentSpace .* mvnpdf(data(:, 1:2), mean2, covariance2);
end

