% n-fold cross validation. Split the data into n sets. Take out (n-1) and
% build classifier and test it using the remaining 1 set. Repeat the
% process for each subset one time.

% If n = number of data, it will become leave-one-out analysis

% Output: evaluation measurements, such as accuracy 

function [avg_accuracy,accuracy_std] = n_fold_cross_validation(fv, label, k , n)
    % parameters
    accuracy = 0;
    num_patients = numel(label);
    factor = round(num_patients/n);
    
    % randomize the data
    rand_pos = randperm(num_patients);
    fv = fv(rand_pos,:);
    label = label(rand_pos);
    
    % n-fold cross validation,e.g. n=10
    % in each fold, calculate the evaluation matrices, and then average
    % everything at the end
    accuracies = zeros(1,n); % to store the list of accuracies for cross validation
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
        for i = 1:num_validation
            pred_validation(i) = kchbox_knn(k,validation_data(i,:),train_data,train_label);        
        end
    
        % Generate evaluation matrice, such as accuracy
        %validation_label
        %pred_validation
        [accuracy] = evaluation_matrice(validation_label,pred_validation);
        
        % update evaluation
        accuracies(nn) = accuracy;
    end
    
    % Calculate average performance
    avg_accuracy = mean(accuracies);
    accuracy_std = std(accuracies);

end