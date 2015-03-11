%% Initialize parameters randomly based on layer sizes. Include weights and biases

function [theta,netconfig] = initializeParameters(inputSize,hiddenLayersSize,numClasses)
    % ################# For hidden layers
    % Choose weights uniformly from the interval [-r, r]
    % Use stack to store the hidden layers parameters, good for minfunc
    % (optimization) and multi-layer neural network implementation
    stack = cell(numel(hiddenLayersSize),1);
    stackparams = [];
    netconfig = [];
    
    %Check if there is any hidden layers. If none, then make the
    %hiddenLayer as the input layer
    if numel(hiddenLayersSize >0)
        % for input layer to first hidden layer        
        r  = sqrt(6) / sqrt(hiddenLayersSize(1) + inputSize + 1); 

        stack{1}.w = rand(hiddenLayersSize(1), inputSize) * 2 * r - r;
        stack{1}.b = zeros(hiddenLayersSize(1), 1);

        % In between hidden layers
        for i = 2:numel(hiddenLayersSize)
            r  = sqrt(6) / sqrt(hiddenLayersSize(i) + hiddenLayersSize(i-1) + 1); 
            
            stack{i}.w = rand(hiddenLayersSize(i), hiddenLayersSize(i-1)) * 2 * r - r;
            stack{i}.b = zeros(hiddenLayersSize(i), 1);       
        end
        [stackparams,netconfig] = stack2params(stack);
    else
        hiddenLayersSize = inputSize;
    end
    
    % ###############For Softmax; no bias
    softmaxW = 0.005 * randn(hiddenLayersSize(end) * numClasses, 1);
 
    
    % ###############stack the parameters by unrolling
    theta = [softmaxW(:) ; stackparams];

end

