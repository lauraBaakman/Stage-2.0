function [ data, uniformDensity ] = uniformDistribution( lower, upper, N, dim )
    %UNIFORMDIST Samples N values from a normal distribution with mean mu
    %and standard deviation sd in dimension dim.
    % INPUT
    %   - lower:   The lowest value allowed
    %   - upper:   The highest value allowed
    %   - N:    Number of samples to be taken
    %   - dim:  Dimension of the distribution
    % OUTPUT
    %   - data: A vector of samples form the distribution
    
   rng(7);
    
    % values
    data = lower + rand(N, dim) .* (upper - lower);
    uniformDensity = 1 / ((upper - lower) ^ dim);        
end

