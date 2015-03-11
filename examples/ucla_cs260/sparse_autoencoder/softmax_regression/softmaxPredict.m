% Prediction using softmax

function [pred] = softmaxPredict(softmaxModel, data)

% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
                                % size:10x8
pred = zeros(1, size(data, 2));  %1x10000

prob = theta * data; %10*10000
[values,pos] = max(prob,[],1);
pred = pos;

end

