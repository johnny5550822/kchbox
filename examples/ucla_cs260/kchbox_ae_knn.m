%% Sparse Autoencoder + knn
% Use the features obtained from sparse autoencoder and then use knn for
% classifier instead of softmax

function [pred] = kchbox_ae_knn(validation_data,train_data, label,...
            numClasses,hiddenSize, sparsityParam,lambda,beta,distance_measure,k)

%--------------  Obtain random parameters theta
inputSize = size(train_data,2);
theta = initializeParameters_ae(hiddenSize, inputSize);

%------------- STEP 2 & 3: Implement sparseAutoencoderCost & Gradient Checking
Debug = false;
if Debug
    [cost, grad] = sparseAutoencoderCost(theta, inputSize, hiddenSize, lambda, ...
                                     sparsityParam, beta, train_data);

    % Check if the the computeNumerical Gradient returns correct values for simple
    % gradients
    checkNumericalGradient();

    % Check your cost function and derivative calculations
    % for the sparse autoencoder.  
    numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, inputSize, ...
                                                      hiddenSize, lambda, ...
                                                      sparsityParam, beta, ...
                                                      train_data), theta);

    % Use this to visually compare the gradients side by side
    disp([numgrad grad]); 

    % Compare numerically computed gradients with the ones obtained from backpropagation
    diff = norm(numgrad-grad)/norm(numgrad+grad);
    disp(diff); % These values are usually less than 1e-9.
end

%----------------------- Step4 Train an autoencoder

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 200;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, train_data), ...
                              theta, options);

%---------------------------- Step 5: Extract features

clc;
trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       train_data');
%size(trainFeatures); 20x15298
%size(trainLabels); 1x15298
testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       validation_data');

%------------------------------- Step 6: Train a Knn classifier
clc;

% With obtained features
trainFeatures = [ones(1,size(trainFeatures,2));trainFeatures];  %21x35
testFeatures = [ones(1,size(testFeatures,2));testFeatures]; %21x4

%------------------------------- Prediction using Knn and the new features
testFeatures = testFeatures';
trainFeatures = trainFeatures';
pred = zeros(1,size(testFeatures,1));
for i = 1:size(testFeatures,1)
    pred(i) = kchbox_knn(k,testFeatures(i,:),trainFeatures,label,distance_measure);      
end
    
    
    
end
