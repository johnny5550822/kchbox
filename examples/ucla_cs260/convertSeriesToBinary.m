% To convert a time-series data points into 0 or 1 depending on whether the
% data points belong to decrease or increase

% Input: mxn, where n is the number of data points, m is number of
% individuals
% output: mx(n-1), because there will be one missing, number of peaks
% (maximum or minimum)

function [states,max_peaks,min_peaks] = convertSeriesToBinary(t_s)
    states = zeros(size(t_s,1),size(t_s,2)-1);    % to store binary 0(decreasing) or 1(increasing)
    max_peaks = zeros(size(t_s,1),1);
    min_peaks = zeros(size(t_s,1),1);
    
    %determine state
    for num = 1:size(t_s,1)
        for i = 1:size(t_s,2)-1
            if (t_s(num,i+1) < t_s(num,i))
                states(num,i) = 0;
            else
                states(num,i) = 1;
            end      
        end
    end

    %determine peak
    for num = 1:size(states,1)
        for i = 1:size(states,2)-1
            if (states(num,i+1)>states(num,i))
                min_peaks(num) = min_peaks(num) + 1;
            elseif (states(num,i+1)<states(num,i))
                max_peaks(num) = max_peaks(num) + 1;
            end
        end
    end

end