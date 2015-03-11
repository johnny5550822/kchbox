% Prediction using NN

%output: pred = prediction; prob = values in softmax, used to plot ROC
%curve

function [pred,prob] = NNPredict(theta, inputSize, hiddenLayersSize, numClasses, netconfig, data)

% Unroll theta parameter
hiddenSize = hiddenLayersSize(end);

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

n = numel(stack);

z=cell(1,n+1);
a=cell(1,n+1);
a{1} = data;
for k = 1:n
   num_col = size(a{k},2);
   z{k+1} = stack{k}.w*a{k} + repmat(stack{k}.b,1,num_col);
   a{k+1} = sigmoid(z{k+1});
end

prob = softmaxTheta * a{n+1};
[values,pos] = max(prob,[],1);
pred = pos;

% -----------------------------------------------------------

end
