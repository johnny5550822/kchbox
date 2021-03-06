% Combine different algorithm tgt and see if it get even better result

%% AE + KNN

%% Step 0 parameter setting

%  change the parameters below.
clear all;
clc;
addpath general_functions/ %add useful functions

% inputSize = 26 * 2; % time-series + two parameters <--define later,
% depends on the feature set
numClasses = 2;
hiddenSizes = {[2],[4],[6],[8],[10],[12],[14],[16],[18],[20]}; % one have one layer since it is an autoencoder, 5-5 is good with 14 features

sparsityParams = [0.1 0.01 0.001];   % desired average activation of the hidden units.                        
lambdas = [3e-2 3e-3 3e-4 3e-5];         % weight decay parameter,3e-3 is the best so far       
beta = 3;              % weight of sparsity penalty term     
numExperiment = 1;

%add functions path
addpath sparse_autoencoder
addpath sparse_autoencoder/minFunc/
addpath sparse_autoencoder/softmax_regression

Debug = true;
if Debug
    hiddenSizes = {[18]}
    sparsityParams = [0.01];   % desired average activation of the hidden units.                        
    lambdas = [0.0003];         % weight decay parameter       
    beta = 3;              % weight of sparsity penalty term  
    
        % Create file to store result
    fileID = fopen('result_temp/ae_knn/result_ae_knn.txt','w');
else    
    % Create file to store result
    fileID = fopen('leave-one-out-result_dataset2/ae_knn/result_ae_knn.txt','w');
end

%% Step 1 Load data

clc;
%read diastolic data, nxm, where m is the number of data point(feature).
%And n is the number of patients
raw_d = xlsread('combine_dia_ldl');   
dia = raw_d(:,2:end);

%read systolic data
raw_s = xlsread('combine_sys_ldl');   
systo = raw_s(:,2:end);

%read label
label = xlsread('label_ldl');
label = label(:,2);

% start label from 1, not 0
label = label + 1;

%% Step 2. Feature generation
fv = generate_features_vector(dia,systo);   % fv = feature vectors

%% Step 3. n-fold cross-validation. If n = number of data, it will become leave-one-out
clc;
n = 39;
k=3;
dist_measure = {'euclidean','minkowski','chebychev'}; % no 'jaccard','hamming' because not suitable for AE

for pp = 1: numel(dist_measure)
    for qq = 1: numel(sparsityParams)
        for kk = 1:numel(lambdas)
            for tt = 1:numel(hiddenSizes)   
                %Select particular parameters
                hiddenSize = hiddenSizes{tt}; % numel(hiddenLayersSize) = number of hidden layers
                lambda = lambdas(kk);         % weight decay parameter             
                sparsityParam = sparsityParams(qq);

                %cross-validation
                [acc,sen,spec,pre,recall,f_measure,mcc,confusion_matrix,true_label] = n_fold_cross_validation_ae_knn(fv,label,n,...
                    numClasses,hiddenSize,sparsityParam,lambda,beta,dist_measure{pp},k); 
                % Euclidean, minkowski, Chebychev are preetty good

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
                % plotROC(probs,true_label);

                 %% Step 4. Write the result to a text file and save figure

                %write to file
                fprintf(fileID,sprintf('############Experiment:%d################\n',numExperiment));
                fprintf(fileID,sprintf('Dist measure: %s \n',dist_measure{pp}));
                fprintf(fileID,sprintf('HiddenLayer:%s\t sparsityParam:%f\t lambda:%f\t beta:%f\n'...
                    ,mat2str(hiddenSize),sparsityParam,lambda,beta));
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
    end
end

%close the file
fclose(fileID);
