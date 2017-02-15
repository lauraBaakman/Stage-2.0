function [data] = normalDistribution(mu, var, N)
    %NORMALDIST Samples N values from a normal distribution with mean mu
    %and standard deviation sd in dimension dim.
    % INPUT
    %   - mu:   The mean(s) of the distribution
    %   - var:   The variance of the distribution
    %   - N:    Number of samples to be taken
    %   - dim:  Dimension of the distribution
    % OUTPUT
    %   - data: A vector of samples form the distribution
    
    % Sample N random vectors from the multivariate normal distribution
    data = mvnrnd(mu,var,N);
    
%     % Compute true values
%     if(dim == 1)
%         % pdf expects a standarddeviation, not variance
%         sd = sqrt(var);
%         trueValues = pdf('normal', data, mu, sd);
%     else
%         trueValues = mvnpdf(data, mu, var);
%     end
end

