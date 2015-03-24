%% softmax classifier
% The functions softmaxCost.m, softmaxPredict.m, softmaxTrain.m are based on Stanford UFLDL tutorial

function [pred,prob,softmaxModel] = kchbox_softmax(testData,trainData, trainData_label,...
    numClasses,inputSize, options,varargin)

    %parameters for function
    if (numel(varargin) < 1 || isempty(varargin{1}))
        softmax_lambda = 1e-4;
    else
        softmax_lambda = varargin{1};
    end    
    % ----------------------------

    %flip data dimension for softmax classifier
    f_trainData = trainData';
    f_testData = testData';
    f_trainData_label = trainData_label';

    % Add bias
    bf_trainData = [ones(1,size(f_trainData,2));f_trainData];
    numFeaturesWithBase = inputSize + 1;
    softmaxModel = softmaxTrain(numFeaturesWithBase, numClasses, softmax_lambda, ...
    bf_trainData,f_trainData_label, options);

    % test
    bf_testData = [ones(1,size(f_testData,2));f_testData];
    [pred, prob] = softmaxPredict(softmaxModel, bf_testData);
end









