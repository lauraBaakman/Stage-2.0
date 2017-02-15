function [ density ] = computeWallDensity(lower, upper)
    %UNTITLED2 Summary of this function goes here
    %   Detailed explanation goes here
    density = 1;
    for i = 1:length(lower)
        density = density / (upper(i) - lower(i));        
    end
end

