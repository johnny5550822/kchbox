% Calculate the mean and std of an array with dimension 1xn.

function [average,standard_deviation] = get_mean_std(data)
    average = mean(data);
    standard_deviation = std(data);
end