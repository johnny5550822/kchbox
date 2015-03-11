% Plot ROC curve of the classifier by varying the threshold

function plotROC(probs,true_label)
    probs = probs(2,:); % only interested in prediction of yes (i.e., 2)

    % Get the values for each class
    positive_pos = find(true_label==2);
    negative_pos = find(true_label==1);

    positive_probs = probs(positive_pos);
    negative_probs = probs(negative_pos);
    
    %Calculate ROC
    max_value = max(positive_probs);
    min_value = min(negative_probs);
    T_roc = max_value:-1:min_value-1; %Threshold for ROC curve
    pos_ROC=[];
    neg_ROC=[];
    for t=T_roc
        pos_ROC = [pos_ROC sum((positive_probs>t))];
        neg_ROC = [neg_ROC sum((negative_probs>t))]; 
    end

    pos_ROC= pos_ROC/length(positive_probs);
    neg_ROC = neg_ROC/length(negative_probs);

    %Plot ROC
    figure;
    y = plot(neg_ROC,pos_ROC,'x-');
    title(strcat('ROC curves for'));
    xlabel('False Alarm');
    ylabel('TPR');

end