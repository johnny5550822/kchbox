% Cost for softmax

function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

%size(data)  %8x100
% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);  %10x8

numCases = size(data, 2); %100
m = numCases;

groundTruth = full(sparse(labels, 1:numCases, 1));  %10x100
cost = 0;

thetagrad = zeros(numClasses, inputSize); %10x8

% calculate cost
hypothesis = calculate_hypothesis(theta,data);  %10x100
each_k = groundTruth.*log(hypothesis); %the inner summation of the cost with respect to k
regularization = lambda/2 * sum(sum(theta.^2));
simple_cost = -1/m*sum(sum(each_k,1));
cost = simple_cost+regularization;

%calculate grad
difference = groundTruth - hypothesis;  %10x100
simple_thetagrad = -1/m*(difference * data');
regularization_grad = lambda * theta; 
thetagrad = simple_thetagrad + regularization_grad;

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

function hypothesis = calculate_hypothesis(theta,data)
    %calculate the denominator of hypothesis
    exp_power = theta * data; %10x8  8x100 -->10x100
    % subract the largest value to avoid possible overflow when exponential
    % is applied.
    exp_power = bsxfun(@minus, exp_power, max(exp_power,[],1)); 
    denominator = sum(exp(exp_power),1) ;   %size=1x100
    
    %calculate hypothesis (must be 10x100<--for each exponential of 10
    %classes, and 100samples)
    hypothesis = exp(exp_power)./repmat(denominator,size(exp_power,1),1); %10x100
end
