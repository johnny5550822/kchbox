% Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% backprop

function [ cost, grad ] = NNCost(theta, inputSize, hiddenLayersSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% theta: trained weights from the autoencoder
% inputSize: the number of input units
% hiddenLayersSize:  hidden layers size
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example

%% Unroll softmaxTheta parameter

lastHiddenSize = hiddenLayersSize(end);

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:lastHiddenSize*numClasses), numClasses, lastHiddenSize);
%size = 2x26

% Extract out the "stack"
stack = params2stack(theta(lastHiddenSize*numClasses+1:end), netconfig);
%e.g. stack{1}.w =size: 20x52
%e.g. stack{1}.b =size: 26x1

% Compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));   
% size(softmaxTheta) = 2x26

stackgrad = cell(size(stack));
for d = 1:numel(stack)
    % for d = 1 (1st layer), stackgrad{1}.w = [20x784], .b=20x1
    % for d = 2 (2nd layer), stackgrad{2}.w = [20x20], .b=20x1
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end
cost = 0; % Need to compute cost

% Additional variables
M = size(data, 1);  %39
groundTruth = full(sparse(labels, 1:M, 1)); % 2x39
%% Perform forward propagation and backpropagation

%----------------------------Step1: Forward propagation
% Parameters initialization
W_b = {};   %Create cell to store all weight
z_i = {};
a_i = {};

%Pre-process; combine both base and the regular parameters
for d = 1:numel(stack)
    W_b{d} = [stack{d}.b stack{d}.w];    
end

% Add bias to data
data = [ones(size(data, 1), 1) data];   %11x785

% First layer of NN, i.e., input
z_i{1} = data';
a_i{1} = data';

% Forward Propagation

% Layer to layer activation
for i = 1:numel(hiddenLayersSize)
   z_i{i+1} = W_b{i} * a_i{i};
   a_i{i+1} = sigmoid(z_i{i+1});   
   
   a_i{i+1} = [ones(1,size(a_i{i},2)) ; a_i{i+1}];
end
% size(a_i{3}) %27x39

% last hidden layer to softmax. Need to do something because I assume no
% bias for softmax
a_i{end} = a_i{end}(2:end,:);   %remove the bias
% z_i{end+1} = softmaxTheta * a_i{end};

% #######calculate cost (CORRECT)
hypothesis = calculate_hypothesis(softmaxTheta,a_i{end});
each_k = groundTruth.*log(hypothesis); %the inner summation of the cost with respect to k
regularization = lambda/2 * sum(sum(softmaxTheta.^2));

simple_cost = -1/M*sum(sum(each_k,1));
cost = simple_cost+regularization;

% #######calculate thetagrad for softmax
difference = groundTruth - hypothesis;  %10x11
%tri_thetagrad = softmaxTheta(:,2:end)' * difference; %20x11; 
simple_thetagrad = -1/M*(difference * a_i{end}');    %10x20
regularization_grad = lambda * softmaxTheta; 
thetagrad = simple_thetagrad + regularization_grad; %10(digit)x20(parameters)

%------------------------------Step2: backpropagation
delta = cell(1,numel(hiddenLayersSize));    %size equal to the number of hidden layers

%for last layer
delta{end} = -(softmaxTheta' *difference).* sigmoidGradient(z_i{end});
% for other layers until the second layer
for i = numel(hiddenLayersSize):-1:2
    delta{i-1} = (W_b{i}(:,2:end)'*delta{i}).*sigmoidGradient(z_i{i});
    
end

%change in gradient
tri = cell(1,numel(hiddenLayersSize)); 
for i = numel(hiddenLayersSize):-1:1

    tri{i} = delta{i} * a_i{i}';
end

%-----------------------------Step3: Update
% Do not need to regularize bias term
theta_grad = {};
for i = 1:numel(tri)
    theta_grad{i} = 1/M * tri{i};
end

% assign value
for i = 1:numel(stackgrad)
    stackgrad{i}.w = theta_grad{i}(:,2:end);
    stackgrad{i}.b = theta_grad{i}(:,1);
end
softmaxThetaGrad = thetagrad;

% ##########################backup, hard code layer
% %----------------------------Step1: Forward propagation
% %Pre-process; combine both base and the regular parameters
% W_b1 = [stack{1}.b stack{1}.w]; %20x785
% W_b2 = [stack{2}.b stack{2}.w]; %20x21
% %size(softmaxTheta)  %10x20
% % data = data';
% data = [ones(size(data, 1), 1) data];   %11x785
% %########Propagation#########
% %1st layer(input) to 2nd layer(1st hidden layer) activation
% z_2 = W_b1*data'; %20*11
% a_2 = sigmoid(z_2); %20*11
% a_2 = [ones(1,size(data,1)) ; a_2]; %21x11
% 
% %2nd layer to 3rd layer(2nd hidden layer) activation
% z_3 = W_b2*a_2; %20x11
% a_3 = sigmoid(z_3); %20x11(20 unit;11 examples)
% % a_3 = [ones(1,size(data,1)) ; a_3]; %21(activiation)x11(sample) <-- for softmax; may not
% % %necessary
% 
% %3rd layer to softmax
% z_4 = softmaxTheta*a_3; %10x11
% %size(z_4)
% % size(softmaxTheta) %10x20
% % size(a_3)
% 
% % #######calculate cost (CORRECT)
% hypothesis = calculate_hypothesis(softmaxTheta,a_3);  %10x11
% each_k = groundTruth.*log(hypothesis); %the inner summation of the cost with respect to k
% regularization = lambda/2 * sum(sum(softmaxTheta.^2));
% simple_cost = -1/M*sum(sum(each_k,1));
% cost = simple_cost+regularization;
% 
% % #######calculate thetagrad for softmax
% difference = groundTruth - hypothesis;  %10x11
% %tri_thetagrad = softmaxTheta(:,2:end)' * difference; %20x11; 
% simple_thetagrad = -1/M*(difference * a_3');    %10x20
% regularization_grad = lambda * softmaxTheta; 
% thetagrad = simple_thetagrad + regularization_grad; %10(digit)x20(parameters)
% 
% %------------------------------Step2: backpropagation
% %size(softmaxTheta' * difference); %20x11
% %size(sigmoidGradient(z_4))  %10x11
% delta_last = -(softmaxTheta' *difference).* sigmoidGradient(z_3); %10(parameters)x11
%         %<-- at layer 3
% delta_2 = (W_b2(:,2:end)'*delta_last).*sigmoidGradient(z_2); %20x11
%         %<--- at layer 2 
% 
% tri_2 = delta_last*a_2';   %20x21  
% tri_1 = delta_2*data;   %20x785
% 
% %-----------------------------Step3: Update
% % DO not regularized the bias term
% Theta1_grad = 1/M*tri_1;  %size=20*785
% Theta2_grad = 1/M*tri_2;  %size=20*21
% 
% % no regularization on the bias term
% % Theta1_grad(:,1) = Theta1_grad(:,1) - lambda*W_b1(:,1);
% % Theta2_grad(:,1) = Theta2_grad(:,1) - lambda*W_b2(:,1);
% 
% %Assign value
% stackgrad{1}.w = Theta1_grad(:,2:end);
% stackgrad{2}.w = Theta2_grad(:,2:end);
% stackgrad{1}.b = Theta1_grad(:,1);
% stackgrad{2}.b = Theta2_grad(:,1);
% softmaxThetaGrad = thetagrad;

% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

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