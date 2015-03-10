% Neural network (with softmax classifier at the end) on a test data
% Input: test_d(mxf), train_d(mxf), train_label(1xm), where f is the number
% of features, m is the number of training data.
% Output: prediction of the test data.


function pred = kchbox_nn(test_d,train_d,train_label,numClasses,hiddenLayersSize,...
    sparsityParam,lambda,beta)

%------------------------ Step 1. NN parameters initialization
% weights include hidden layers, the softmax classifer weight, and the
% biases. The network structure is 52x26x26x2
inputSize = size(train_d,2); %input size for neural network, i.e. 52
[theta,netconfig] = initializeParameters(inputSize,hiddenLayersSize,numClasses);


%------------------------- Step 2. Train the neural network with a softmax
%(assume no bias in softmax)

%------------- Step 2.1 Check the cost function and see if it is correct
[cost,grad] = NNCost(theta,inputSize,hiddenLayersSize,numClasses,...
     netconfig,lambda,train_d,train_label);

DEBUG = false;
if DEBUG
    checkNNCost;
end

%------------- Step 2.2 optimize the theta using minfunc (which is a 
%gradient descent algorithm)

%  Use minFunc to minimize the function
options.HessUpate = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.MaxIter = 100;	  % Maximum number of iterations of L-BFGS to run 
options.Display = 'iter';
options.GradObj = 'on';

[optTheta, cost] = minFunc( @(p) NNCost(p, ...
                                   inputSize, hiddenLayersSize, ...
                                   numClasses, netconfig, ...
                                   lambda, train_d, train_label), ...
                              theta, options);

%--------------- Step 3: Test
[pred] = NNPredict(optTheta, inputSize, hiddenLayersSize, ...
                          numClasses, netconfig, test_d');                       

% evaluation
% acc = mean(train_label(:) == pred(:));
% fprintf('Accuracy: %0.3f%%\n', acc * 100);



end






