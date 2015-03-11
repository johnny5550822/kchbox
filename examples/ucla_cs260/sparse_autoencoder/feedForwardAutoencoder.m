% Feed forward propagation

function [activation] = feedForwardAutoencoder(theta, hiddenSize, visibleSize, data)
activation =[];
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize); %20x784
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize); %20x1

%Combine W1 and b1
W_b1 = [b1 W1]; %20x785
% trainData = 784x15298
% testData = 784x15298

data = [ones(1,size(data,2));data]; %Constant 1 in 1st row for base term
    % 785x15298
activation = sigmoid(W_b1*data); %20x15298;



end

