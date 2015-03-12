% Returns a confusion matrix

function [confusion_matrix] = get_confusion_matrix(true_label,pred_label)
    %parameters
    num_data = numel(true_label);
    tp = 0;
    tn = 0;
    fp = 0;
    fn = 0;
    
    for i = 1:num_data
       if (true_label(i) == pred_label(i))
            if (true_label(i)==1)
                tp = tp + 1;
            else
                tn = tn + 1;
            end
       else
           if (true_label(i)==1)
               fn = fn + 1;
           else
               fp = fp + 1;
           end
       end        
    end
    
    %matrices
    accuracy = (tp+tn)/num_data;
    sensitivity = tp/(tp+fn);
    specificity = tn/(tn+fp);
    precision = tp/(tp+fp);
    recall = tp/(tp+fn);
    f_measure = 2*(precision*recall)/(precision+recall);
    confusion_matrix = [tp fp;fn tn];
    mcc = (tp*tn-fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));
    %roc %<-- this depends on the classifier and have to change the
    %threshold    
end




