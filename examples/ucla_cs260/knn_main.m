%% Created by King Chung Ho (Johnny Ho) in 030515
% For UCLA CS260 project (Winter 2015)
% Objective: Use kchbox-knn to build a model for prediction
% Feature: 2 features (Diastolic and systolic) along different time
% Number of training data: 39 patients, 20 with label 0, and 19 with label
% 1

% WIll attempt different method and see which one give the best result.
% Have a feature generation function to generate different features
%% Step 0 Creating training data (random) for 2 clusters using gaussian distribution.
clear all;
clc;
addpath data/

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
label = label+1;

% NOTE: One possible step here is to normalize the data series for each
% patient. However, the hypoethesis is that the magnitude of the pressure
% matter relative to different type of patients. So, I did not perform this
% step
%% *********************METHOD 1 Simple knn with Euclidean distance*******

%% Step 1. Feature generation
fv = generate_features_vector(dia,systo);   % fv = feature vectors

%% Step 2 Define k + parameter setting
k = 3;

%% Step 3 & Step 4
split_ratio = 0.7;
% [accuracy] = simple_validation(fv,label,split_ratio,k);
%% n-fold cross-validation. If n = number of data, it will become leave-one-out
numExperiment = 1;

fileID = fopen('result/knn/result_knn.txt','w');
n = 10;
dist_measure = {'euclidean','minkowski','chebychev','jaccard','hamming'};

for i = 1:numel(dist_measure)
    for k = 2:5
    [acc,sen,spec,pre,recall,f_measure,mcc,confusion_matrix] = n_fold_cross_validation_knn(fv,label,k,n,dist_measure{i});
        disp(sprintf('confusion_matrix:'));
        disp(confusion_matrix)
        disp(sprintf('Accuracy:%f',acc));
        disp(sprintf('Sentivity:%f',sen));
        disp(sprintf('Specificity:%f',spec));
        disp(sprintf('Precision:%f',pre));
        disp(sprintf('Recall:%f',recall));
        disp(sprintf('F-measure:%f',f_measure));
        disp(sprintf('MCC:%f',mcc));
        
        
        %% Step 4. Write the result to a text file and save figure

        %write to file
        fprintf(fileID,sprintf('############Experiment:%d################\n',numExperiment));
        fprintf(fileID,sprintf('k:%f \t dist_measure:%s \t \n',k,dist_measure{i}));
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
%% DONE!

