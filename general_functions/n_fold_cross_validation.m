% n-fold cross validation. Split the data into n sets. Take out (n-1) and
% build classifier and test it using the remaining 1 set. Repeat the
% process for each subset one time.

% If n = number of data, it will become leave-one-out analysis

% Input: accept multiple test data in matrix

% Output: evaluation measurements, such as accuracy 

% Note, the output is the combined statistics (not average) based on a
% combined confusion matric

% varargin{1} = options; for minfunc(the gradient descent algorithm)
% varargin{2} = softmax_lambda; for softmax
% varargin{3} = sparsityParam; for autencoder
% varargin{4} = lambda; for autoencoder
% varargin{5} = beta; for autoecnoder

function [acc,sen,spec,pre,recall,f_measure,mcc,confusion_matrix,probs,label] ...
    = n_fold_cross_validation_ae(classifier_type,fv, label, n, numClasses,inputSize,...
    varargin)

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
    % parameters
    num_patients = numel(label);
    factor = round(num_patients/n);
    
    % randomize the data
    rand_pos = randperm(num_patients);
    fv = fv(rand_pos,:);
    label = label(rand_pos);
    
    % n-fold cross validation,e.g. n=10
    % in each fold, calculate the evaluation matrices, and then average
    % everything at the end
    confusion_matrice = zeros(2,2,n);
    probs = []; %store the values generated by NN; used for ROC curve
    for nn = 1:n
        nn;
        %########################Obtain data
        if (nn*factor < num_patients)
            cutpoint = factor*nn;
        else               
            cutpoint = num_patients;
        end
        % validation data 
        valid_pos = (factor*(nn-1))+1:cutpoint;
        validation_data = fv(valid_pos,:);
        validation_label = label(valid_pos);
            
        % train data
        train_pos = [1:(factor*(nn-1)) cutpoint+1:num_patients];
        train_data = fv(train_pos,:);
        train_label = label(train_pos);   
        
        % prediction
        [num_validation] = size(validation_data,1);
        pred_validation = zeros(1,num_validation);        
        if strcmp(classifier_type,'softmax')
            [pred_validation,prob] = kchbox_softmax(validation_data,train_data,train_label,...
                numClasses,inputSize,options,softmax_lambda,sparsityParam,lambda,beta);
        else if strcmp(classifier_type,'sparse_ae')
            [pred_validation,prob] = kchbox_ae(validation_data,train_data,train_label,...
                numClasses,inputSize,options,softmax_lambda,sparsityParam,lambda,beta);
        end
        
        % Generate evaluation matrice, such as accuracy. Also, for ROC
        %validation_label
        %pred_validation
        [confusion_matrix] = get_confusion_matrix(validation_label,pred_validation);
        probs = [probs prob];
        
        % update evaluation
        confusion_matrice(:,:,nn) = confusion_matrix;
    end

    %Calculate evaluation measures
    confusion_matrix = sum(confusion_matrice,3);
    [acc,sen,spec,pre,recall,f_measure,mcc,confusion_matrix] = ...
        get_evaluation_matrice(confusion_matrix);

end