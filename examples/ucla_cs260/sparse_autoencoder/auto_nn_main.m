%% Sparse Autoencoder which can be used as dimenion reduction in a non-linear fashion

% NOTE: make sure the features are nomalized to 0 and 1 (for autoencoder)
%% Step 0 parameter setting

%  change the parameters below.
clear all;
clc;

% inputSize = 26 * 2; % time-series + two parameters <--define later,
% depends on the feature set
numClasses = 2;
hiddenSize = [5]; % one have one layer since it is an autoencoder

sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-5;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term       

%add functions path
addpath sparse_autoencoder
addpath sparse_autoencoder/minFunc/

%% Step 1 Load data

clc;
%read diastolic data, nxm, where m is the number of data point(feature).
%And n is the number of patients
raw_d = xlsread('combine_dia');   
dia = raw_d(:,2:end);

%read systolic data
raw_s = xlsread('combine_sys');   
systo = raw_s(:,2:end);

%read label
label = xlsread('label');
label = label(:,2);

% start label from 1, not 0
label = label + 1;

%% Step 2. Feature generation
fv = generate_features_vector(dia,systo);   % fv = feature vectors

%  Obtain random parameters theta
inputSize = size(fv,2);
theta = initializeParameters_sa(hiddenSize, inputSize);

%%======================================================================
%% STEP 2 & 3: Implement sparseAutoencoderCost & Gradient Checking
clc;

Debug = false;
if Debug
    [cost, grad] = sparseAutoencoderCost(theta, inputSize, hiddenSize, lambda, ...
                                     sparsityParam, beta, fv);

    % Check if the the computeNumerical Gradient returns correct values for simple
    % gradients
    checkNumericalGradient();

    % Check your cost function and derivative calculations
    % for the sparse autoencoder.  
    numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, inputSize, ...
                                                      hiddenSize, lambda, ...
                                                      sparsityParam, beta, ...
                                                      fv), theta);

    % Use this to visually compare the gradients side by side
    disp([numgrad grad]); 

    % Compare numerically computed gradients with the ones obtained from backpropagation
    diff = norm(numgrad-grad)/norm(numgrad+grad);
    disp(diff); % These values are usually less than 1e-9.
end
%%======================================================================
%% Step4 Train an autoencoder

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 100;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, fv), ...
                              theta, options);

%%======================================================================
%% Step 5: Extract features

clc;
trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       fv');
%size(trainFeatures); 20x15298
%size(trainLabels); 1x15298
testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       fv');

%% Step 6: Train a softmax classifier
clc;
options.maxIter = 100;
addpath sparse_autoencoder/softmax_regression
softmax_lambda = 1e-4;

% With obtained features
trainFeatures = [ones(1,size(trainFeatures,2));trainFeatures];  %for base term
hiddenSizeWithBase = hiddenSize+1;
softmaxModel = softmaxTrain(hiddenSizeWithBase, numClasses, softmax_lambda, ...
                            trainFeatures, label, options);
%% Step 7: Test
%Assign test features (activation) to inputData
inputData = testFeatures;
% inputData = testData;
inputData = [ones(1,size(inputData,2));inputData]; %for base

% You will have to implement softmaxPredict in softmaxPredict.m
[pred] = softmaxPredict(softmaxModel, inputData);

% Classification Score
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == label(:)));