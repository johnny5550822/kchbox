%% Sparse Autoencoder which can be used as dimenion reduction in a non-linear fashion

function [pred,prob] = kchbox_ae(validation_data,train_data,label,...
            numClasses,hiddenSize, varargin)
        
    %parameters for function
    if (numel(varargin) < 1 || isempty(varargin{1}))
        options.maxIter = 400;
        options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
        options.display = 'on';
    else
        options = varargin{1};
    end
    if (numel(varargin) < 2 || isempty(varargin{2}))
        softmax_lambda = 1e-4;   
    else
        softmax_lambda = varargin{2};
    end
    if (numel(varargin) < 3 || isempty(varargin{3}))
        sparsityParam = 0.05;   % desired average activation of the hidden units.  
    else
        sparsityParam = varargin{3};
    end
    if (numel(varargin) < 4 || isempty(varargin{4}))
        lambda = 3e-5;         % weight decay parameter         
    else
        lambda = varargin{4};
    end
    if (numel(varargin) < 5 || isempty(varargin{5}))
        beta = 3;              % weight of sparsity penalty term 
    else
        beta = varargin{5};
    end

    
    %----------------------------------------------------------------------

        
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

%------------------------------- Step 6: Train a softmax classifier
clc;

% With obtained features
trainFeatures = [ones(1,size(trainFeatures,2));trainFeatures];  %for base term
hiddenSizeWithBase = hiddenSize+1;
softmaxModel = softmaxTrain(hiddenSizeWithBase, numClasses, softmax_lambda, ...
                            trainFeatures, label, options);
%--------------------------------- Step 7: Test
% inputData = testData;
testFeatures = [ones(1,size(testFeatures,2));testFeatures]; %for base

% You will have to implement softmaxPredict in softmaxPredict.m
[pred, prob] = softmaxPredict(softmaxModel, testFeatures);

end
