%% Initialize parameters randomly based on layer sizes. Include weights and biases

function theta = initializeParameters(inputSize,hiddenLayersSize,numClasses)
    % ################# For hidden layers
    % Choose weights uniformly from the interval [-r, r]
    W = {};
    b = {};
    
    %Check if there is any hidden layers. If none, then make the
    %hiddenLayer as the input layer
    if numel(hiddenLayersSize >0)
        % for input layer to first hidden layer        
        r  = sqrt(6) / sqrt(hiddenLayersSize(1) + inputSize + 1); 
        W{1} = rand(hiddenLayersSize(1), inputSize) * 2 * r - r;
        b{1} = zeros(hiddenLayersSize(1), 1);

        % In between hidden layers
        for i = 2:numel(hiddenLayersSize)
            r  = sqrt(6) / sqrt(hiddenLayersSize(i) + hiddenLayersSize(i-1) + 1); 

            W{i} = rand(hiddenLayersSize(i-1), hiddenLayersSize(i)) * 2 * r - r;

            b{i} = zeros(hiddenLayersSize(i), 1);
        end
    else
        hiddenLayersSize = inputSize;
    end

    % ###############For Softmax; no bias
    softmaxW = 0.005 * randn(hiddenLayersSize(end) * numClasses, 1);

    % Convert weights and bias gradients to the vector form, unrolling
    theta = [softmaxW(:)];
    
    for i = 1:numel(W)
        this_W = W{i}; 
        theta = [theta ; this_W(:)];
    end
    for i = 1:numel(b)
        this_b = b{i}; 
        theta = [theta ; this_b(:)];
    end
end

