function [ data, densities, Ns] = ferdosi_5( N, d)
    %FERDOSI_5 Dataset one from ferdosi et al.
    % [DATA, DENSITIES, RATIOS] = FERDOSI_2(N, D) generates a D dimensional
    % dataset with N patterns. Where DATA is a N x D matrix where each row
    % is a pattern, DENSITIES is is a N x 1 vector with the true
    % densities of the patterns.
    
    %% INIT
    ratios = [1/3, 1/3, 1/3];
    Ns = ceil(N * ratios);

    clear rows;
    [startRow, endRow] = rows(Ns);
    
    %% Gaussian
    means(:,:,1) = 10;
    covariances(:,:,1) = 5;
    data(startRow:endRow, 2) = normalDistribution(means(:,:,1), covariances(:,:,1), Ns(1));
    
    %% Wall 1
    lower(:,:,1) = [0, 0];
    upper(:,:,1) = [100, 100];
    data(startRow:endRow, [1,3]) = blockUniformDistribution(lower(:,:,1), upper(:,:,1), Ns(1), 2);

    [startRow, endRow] = rows(Ns);
    
    %% Wall 2
    means(:,:,2) = 50;
    covariances(:,:,2) = 5;
    data(startRow:endRow, 3) = normalDistribution(means(:,:,2), covariances(:,:,2), Ns(2));
    
    lower(:,:,2) = [0, 0];
    upper(:,:,2) = [100, 100];
    data(startRow:endRow, [1,2]) = blockUniformDistribution(lower(:,:,2), upper(:,:,2), Ns(2), 2);    
    
    [startRow, endRow] = rows(Ns);
    
    %% Wall 3
    means(:,:,3) = 50;
    covariances(:,:,3) = 5;
    data(startRow:endRow, 2) = normalDistribution(means(:,:,3), covariances(:,:,3), Ns(3));
    
    lower(:,:,3) = [0, 0];
    upper(:,:,3) = [100, 100];
    data(startRow:endRow, [1,3]) = blockUniformDistribution(lower(:,:,3), upper(:,:,3), Ns(3), 2);        
    
    %% WallDensities and Spaces
    wallSpaces(:,:,1) = computeWallSpace(data(:, [1,3]), lower(:,:,1), upper(:,:,1));
    wallDensities(:,:,1) = computeWallDensity(lower(:,:,1), upper(:,:,1));
    
    wallSpaces(:,:,2) = computeWallSpace(data(:, [1,2]), lower(:,:,2), upper(:,:,2));
    wallDensities(:,:,2) = computeWallDensity(lower(:,:,2), upper(:,:,2));    

    wallSpaces(:,:,3) = computeWallSpace(data(:, [1,3]), lower(:,:,3), upper(:,:,3));
    wallDensities(:,:,3) = computeWallDensity(lower(:,:,3), upper(:,:,3));    
    
    densities = ratios(1) * wallDensities(:,:,1) .* wallSpaces(:,:,1) .* mvnpdf(data(:,2), means(:,:,1), covariances(:,:,1)) + ...
                ratios(2) * wallDensities(:,:,2) .* wallSpaces(:,:,2) .* mvnpdf(data(:,3), means(:,:,2), covariances(:,:,2)) + ...        
                ratios(3) * wallDensities(:,:,3) .* wallSpaces(:,:,3) .* mvnpdf(data(:,2), means(:,:,3), covariances(:,:,3));
end