%% Created by King Chung Ho (Johnny Ho) in 03/09/15
% For UCLA CS260 project (Winter 2015)
% Objective: Use neural network(multi-layer) to build a model for prediction
% Feature: 2 features (Diastolic and systolic) along different time
% Number of training data: 39 patients, 20 with label 0, and 19 with label
% 1

% Train a deep neural network (2 layers) with a softmax attached to the
% output nodes

% Note: may add pre-training step to improve performance

%% Step 0 Parameters set-up

%  change the parameters below.
clear all;
clc;

% inputSize = 26 * 2; % time-series + two parameters <--define later,
% depends on the feature set
numClasses = 2;
hiddenLayersSizes = {[5],[10],[20],[5 5],[10 5],[10 10],[20 5],...
    [20 10], [20 20]}; % numel(hiddenLayersSize) = number of hidden layers

lambdas = [5e-3 1e-3 5e-4 1e-4 5e-5 1e-5];         % weight decay parameter 
sparsityParam = 0.1;   % FOR AE, desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
beta = 3;              % FOR AE, weight of sparsity penalty term     

numExperiment = 1; %the experiment number

%add functions path
addpath neural_network
addpath neural_network/minFunc/


%For debug
Debug = true;
if Debug
    hiddenLayersSizes = {[20]}; % numel(hiddenLayersSize) = number of hidden layers
    lambdas = [5e-5];         % weight decay parameter 
 
    % Create file to store result
    fileID = fopen('result/nn/debug_result_nn.txt','w');
else    
    % Create file to store result
    fileID = fopen('result/nn/result_nn.txt','w');
end
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

%% Step 3. n-fold cross-validation. If n = number of data, it will become leave-one-out
clc;

for kk = 1:numel(lambdas)
    for tt = 1:numel(hiddenLayersSizes)   
        %Select particular parameters
        hiddenLayersSize = hiddenLayersSizes{tt}; % numel(hiddenLayersSize) = number of hidden layers
        lambda = lambdas(kk);         % weight decay parameter             

        % Cross validation
        n = 10;
        [acc,sen,spec,pre,recall,f_measure,mcc,confusion_matrix,probs,true_label] = n_fold_cross_validation_nn(fv,label,n,...
            numClasses,hiddenLayersSize,sparsityParam,lambda,beta);

        disp(sprintf('confusion_matrix:'));
        disp(confusion_matrix)
        disp(sprintf('Accuracy:%f',acc));
        disp(sprintf('Sentivity:%f',sen));
        disp(sprintf('Specificity:%f',spec));
        disp(sprintf('Precision:%f',pre));
        disp(sprintf('Recall:%f',recall));
        disp(sprintf('F-measure:%f',f_measure));
        disp(sprintf('MCC:%f',mcc));

        % Plot ROC curve based on the values generated by NN
        h_roc = plotROC(probs,true_label);


        %% Step 4. Write the result to a text file and save figure

        %save figure
        saveas(h_roc,sprintf('result/nn/nn_roc_exp%d.jpg',numExperiment));

        %write to file
        fprintf(fileID,sprintf('############Experiment:%d################\n',numExperiment));
        fprintf(fileID,sprintf('HiddenLayer:%s\t sparsityParam:%f\t lambda:%f\t beta:%f\n'...
            ,mat2str(hiddenLayersSize),sparsityParam,lambda,beta));
        fprintf(fileID,sprintf('confusion_matrix:%s\n',mat2str(confusion_matrix)));
        fprintf(fileID,sprintf('Accuracy:%f\n',acc));
        fprintf(fileID,sprintf('Sentivity:%f\n',sen));
        fprintf(fileID,sprintf('Specificity:%f\n',spec));
        fprintf(fileID,sprintf('Precision:%f\n',pre));
        fprintf(fileID,sprintf('Recall:%f\n',recall));
        fprintf(fileID,sprintf('F-measure:%f\n',f_measure));
        fprintf(fileID,sprintf('MCC:%f\n',mcc));
        fprintf(fileID,'\n');

        %udpate counter
        numExperiment = numExperiment +1;
    end
end


%close the file
fclose(fileID);

