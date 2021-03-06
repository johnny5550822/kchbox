% n-fold cross validation for autoencoder. Split the data into n sets. Take out (n-1) and
% build classifier and test it using the remaining 1 set. Repeat the
% process for each subset one time.

% If n = number of data, it will become leave-one-out analysis

% Input: accept multiple test data in matrix

% Output: evaluation measurements, such as accuracy 

% Note, the output is the combined statistics (not average) based on a
% combined confusion matric

function [acc,sen,spec,pre,recall,f_measure,mcc,confusion_matrix,label] ...
    = n_fold_cross_validation_ae_knn(fv, label, n, numClasses,hiddenLayersSize,...
    sparsityParam,lambda,beta,dist_measure,k)

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
        [pred_validation] = kchbox_ae_knn(validation_data,train_data,train_label,...
            numClasses,hiddenLayersSize, sparsityParam,lambda,beta,dist_measure,k);
        
        % Generate evaluation matrice, such as accuracy. Also, for ROC
        %validation_label
        %pred_validation
        [confusion_matrix] = get_confusion_matrix(validation_label,pred_validation);
        
        % update evaluation
        confusion_matrice(:,:,nn) = confusion_matrix;
    end

    %Calculate evaluation measures
    confusion_matrix = sum(confusion_matrice,3);
    [acc,sen,spec,pre,recall,f_measure,mcc,confusion_matrix] = ...
        get_evaluation_matrice(confusion_matrix);

end