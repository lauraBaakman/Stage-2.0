function [ space ] = computeWallSpace( data, lower, upper )
    %UNTITLED3 Summary of this function goes here
    %   Detailed explanation goes here
    space = ones(size(data,1), 1);
    for i  = 1:length(lower)
        space(:) = space(:) & ...
                data(:,i) < upper(i) &...
                data(:,i) > lower(i);
    end
    
end

