function [ startRow, endRow ] = rows(Ns)
    %UNTITLED4 Summary of this function goes here
    %   Detailed explanation goes here
    persistent index
    if(isempty(index))
        index = 1;
        startRow = 1;
        endRow = startRow + Ns(index) - 1;
    else
        index = index + 1;
        startRow = sum(Ns(1:(index - 1))) + 1;
        endRow = sum(Ns(1:index));
    end
end

