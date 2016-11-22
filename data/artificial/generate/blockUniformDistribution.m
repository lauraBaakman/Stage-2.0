function [ data ] = blockUniformDistribution( lower, upper, N, dim )
    %BLOCKUNIFORMDIST Samples N values from a uniform distribution with 
    % dim dimensions.
    % INPUT
    %   - lower:   The lowest value allowed (per dimension)
    %   - upper:   The highest value allowed (per dimension)
    %   - N:    Number of samples to be taken
    %   - dim:  Dimension of the distribution
    % OUTPUT
    %   - data: A vector of samples form the distribution
 
  % values
  data = rand(N, dim); 
   for i= 1 : dim
    data(:,i) = lower(i) + data(:,i) .* (upper(i) - lower(i));
   end 
end

