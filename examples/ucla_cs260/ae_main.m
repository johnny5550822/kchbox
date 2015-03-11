%% Created by King Chung Ho (Johnny Ho)
% For UCLA CS260 project (Winter 2015)
% Objective: Use autoencoder to build a model for prediction
% Feature: 2 features (Diastolic and systolic) along different time
% Number of training data: 39 patients, 20 with label 0, and 19 with label
% 1

% Only one layer

%% Step 0 parameter setting

%  change the parameters below.
clear all;
clc;

% inputSize = 26 * 2; % time-series + two parameters <--define later,
% depends on the feature set
numClasses = 2;
hiddenSize = [5]; % one have one layer since it is an autoencoder

sparsityParam = 0.1;   % desired average activation of the hidden units.                        
lambda = 3e-5;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term       

%add functions path
addpath sparse_autoencoder
addpath sparse_autoencoder/minFunc/
addpath sparse_autoencoder/softmax_regression

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
n = 10;
[acc,sen,spec,pre,recall,f_measure,mcc,confusion_matrix,probs,true_label] = n_fold_cross_validation_ae(fv,label,n,...
    numClasses,hiddenSize,sparsityParam,lambda,beta);

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
plotROC(probs,true_label);

 
