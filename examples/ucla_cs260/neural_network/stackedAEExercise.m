%% CS294A/CS294W Stacked Autoencoder Exercise

%???????????There are some problems in the stackedAECost????

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  sstacked autoencoder exercise. You will need to complete code in
%  stackedAECost.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises. You will need the initializeParameters.m
%  loadMNISTImages.m, and loadMNISTLabels.m files from previous exercises.
%  
%  For the purpose of completing the assignment, you do not need to
%  change the code in this file. 
%
%%======================================================================
%% STEP 1: Load data from the MNIST database
%
%  This loads our training data from the MNIST database files.
clc;
% Load MNIST database files
trainData = loadMNISTImages('mnist/train-images.idx3-ubyte');
trainLabels = loadMNISTLabels('mnist/train-labels.idx1-ubyte');

trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1

% For debug
DEBUG = true;
if DEBUG
    numEg = 11;
    trainData = trainData(:,1:numEg); 
    trainLabels = trainLabels(1:numEg);
    hiddenSizeL1 = 20;    % Layer 1 Hidden Size
    hiddenSizeL2 = 20;    % Layer 2 Hidden Size
end
%%======================================================================
%% STEP 2: Train the first sparse autoencoder
%  This trains the first sparse autoencoder on the unlabelled STL training
%  images.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.
clc;
addpath sparse_autoencoder/
addpath minFunc/

%  Randomly initialize the parameters
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);


%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the first layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL1"
%                You should store the optimal parameters in sae1OptTheta
clc;

%  Use minFunc to minimize the function
options.HessUpate = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.MaxIter = 10;	  % Maximum number of iterations of L-BFGS to run 
options.Display = 'iter';
options.GradObj = 'on';


[sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSizeL1, ...
                                   lambda, sparsityParam, ...
                                   beta, trainData), ...
                              sae1Theta, options);
                          
%% Visulization of hidden layer 1
clc;
W1 = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), hiddenSizeL1, inputSize);
size(W1)
display_network(W1', 12); 

% -------------------------------------------------------------------------

%%======================================================================
%% STEP 2: Train the second sparse autoencoder
%  This trains the second sparse autoencoder on the first autoencoder
%  featurse.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.
clc
addpath self_taught_learning
[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);
                                    
%  Randomly initialize the parameters
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);

%% Testing: View sae1Features
randnums = randperm(size(sae1Features,2));
display_network(trainData(:,randnums(1:100)));
figure
display_network(sae1Features(:,randnums(1:100)));
%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the second layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL2" and an inputsize of
%                "hiddenSizeL1"
%
%                You should store the optimal parameters in sae2OptTheta

clc;

%  Use minFunc to minimize the function
options.HessUpate = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.MaxIter = 10;	  % Maximum number of iterations of L-BFGS to run 
options.Display = 'iter';
options.GradObj = 'on';


[sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   hiddenSizeL1, hiddenSizeL2, ...
                                   lambda, sparsityParam, ...
                                   beta, sae1Features), ...
                              sae2Theta, options);
                         
% -------------------------------------------------------------------------
%% Visulization of hidden layer 2
clc;
W2 = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), hiddenSizeL2, hiddenSizeL1);
display_network(W2'); 


%%======================================================================
%% STEP 3: Train the softmax classifier
%  This trains the sparse autoencoder on the second autoencoder features.
%  If you've correctly implemented softmaxCost.m, you don't need
%  to change anything here.
clc;
[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);

%  Randomly initialize the parameters
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);
%% Testing : View features
randnums = randperm(size(sae2Features,2));
display_network(sae2Features(:,randnums(1:100)));

%% Testing: save the features so that you don't have to re-run it again
save('optFeatures.mat','sae1OptTheta','sae2OptTheta')
%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the softmax classifier, the classifier takes in
%                input of dimension "hiddenSizeL2" corresponding to the
%                hidden layer size of the 2nd layer.
%
%                You should store the optimal parameters in saeSoftmaxOptTheta 
%
%  NOTE: If you used softmaxTrain to complete this part of the exercise,
%        set saeSoftmaxOptTheta = softmaxModel.optTheta(:);

clc;
options.maxIter = 10;
addpath softmax_regression
softmax_lambda = 1e-4;

%sae2Features = [ones(1,size(sae2Features,2));sae2Features];  %for base term
%hiddenSizeL2WithBase = hiddenSizeL2+1; %for base
softmaxModel = softmaxTrain(hiddenSizeL2, numClasses, softmax_lambda, ...
                            sae2Features, trainLabels, options);

saeSoftmaxOptTheta = softmaxModel.optTheta(:);
% -------------------------------------------------------------------------

%%======================================================================
%% STEP 5: Finetune softmax model

% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.
clc;
% Initialize the stack using the parameters learned
stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the deep network, hidden size here refers to the '
%                dimension of the input to the classifier, which corresponds 
%                to "hiddenSizeL2".
%
%
clc;

%lambda = 0.003;
%assume no base in softmax
[cost,grad] = stackedAECost(stackedAETheta,inputSize,hiddenSizeL2,numClasses,...
    netconfig,lambda,trainData,trainLabels);
%% Check the stacked AE Cost
clc;
DEBUG = true;
if DEBUG
    checkStackedAECost;
end

% -------------------------------------------------------------------------
%% optimize the theta
clc;
 [stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost(p, ...
                                   inputSize, hiddenSizeL2, ...
                                   numClasses, netconfig, ...
                                   lambda, trainData, trainLabels), ...
                              stackedAETheta, options);
%%======================================================================
%% STEP 6: Test 
%  Instructions: You will need to complete the code in stackedAEPredict.m
%                before running this part of the code
%
clc;

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set
testData = loadMNISTImages('mnist/t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('mnist/t10k-labels.idx1-ubyte');

testLabels(testLabels == 0) = 10; % Remap 0 to 10

[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

% Accuracy is the proportion of correctly classified images
% The results for our implementation were:
%
% Before Finetuning Test Accuracy: 87.7%
% After Finetuning Test Accuracy:  97.6%
%
% If your values are too low (accuracy less than 95%), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)

% I obtained Before Finetuning Test Accuracy: 91.980%
%            After Finetuning Test Accuracy: 97.980%